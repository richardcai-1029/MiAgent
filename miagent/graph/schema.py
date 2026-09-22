"""任务图的结构化 schema。

★ 这里是「任务图长什么样」的唯一事实来源：

    Pydantic 模型 ──model_json_schema()──→ 喂给模型的格式说明
                  ──model_validate()────→ 校验模型的输出

  提示词里的格式描述与实际校验逻辑若是两份拷贝，改一处忘另一处
  就会出现「说要 tasks、却校验 steps」这类问题。
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from . import dataflow

_ARGUMENTS_DESCRIPTION = ('工具参数。要用到上游任务的结果时，把该参数的值写成 '
                          '{"$from": "上游任务 id"}，该引用会自动产生依赖关系')


def _reject_stringified_references(arguments: dict[str, Any]) -> dict[str, Any]:
    """引用写成字符串的计划在校验阶段即被拒，错误原因会喂回给模型自修复。

    放在 schema 校验而不是 dag.validate：这一层的失败会触发自修复重试，
    模型有一次改正的机会；到了结构校验就只能整份计划作废。
    """
    bad = dataflow.stringified_references(arguments)
    if bad:
        raise ValueError(
            f"参数 {'、'.join(bad)} 的引用写成了字符串，"
            f'应写成 JSON 对象 {{"{dataflow.REFERENCE_KEY}": "任务 id"}}')
    return arguments


class TaskSpec(BaseModel):
    """模型产出的单个任务。注意它不含 status/result/error ——
    那些是运行时状态，由框架维护，不该让模型填。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="任务唯一标识，如 task_1")
    description: str = Field(description="这一步要达成什么")
    dependencies: list[str] = Field(
        default_factory=list,
        description="必须先完成的任务 id 列表；没有依赖则为空数组")
    required_tool: str = Field(description="要调用的工具名")
    arguments: dict[str, Any] = Field(default_factory=dict, description=_ARGUMENTS_DESCRIPTION)

    _no_stringified_references = field_validator("arguments")(_reject_stringified_references)


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[TaskSpec] = Field(
        description="任务列表；任务间通过 dependencies 表达先后关系，"
                    "无依赖关系的任务可并行执行。无需工具时为空数组")


class FinalOutput(BaseModel):
    """收尾节点的产出。一次调用同时给出回答与本轮摘要：
    回答给用户，摘要给下一轮规划 —— 不为摘要多烧一轮端侧推理。"""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="给用户的回答。简洁，只说结论，不复述过程")
    summary: str = Field(description="本轮摘要：具体结论是什么（含时间、数值等要点），"
                                     "供下一轮规划参考，一两句话")


def task_plan_model_for(tool_names: Sequence[str]) -> type[BaseModel]:
    """按当前可用工具生成收紧的模型：工具名成为 schema 里的枚举。

    收益有二：模型看到的 schema 直接列出合法工具名，减少幻觉；
    万一仍然幻觉，在解析阶段即被拒（AG-1001），不必白跑一步。
    将来接约束解码时，这份 schema 可直接转成采样语法。
    """
    if not tool_names:
        return TaskPlan

    tool_field = Literal[tuple(tool_names)]  # type: ignore[valid-type]
    spec = create_model(
        "TaskSpecConstrained",
        __config__=ConfigDict(extra="forbid"),
        id=(str, Field(description="任务唯一标识，如 task_1")),
        description=(str, Field(description="这一步要达成什么")),
        dependencies=(list[str], Field(default_factory=list,
                                       description="必须先完成的任务 id 列表")),
        required_tool=(tool_field, Field(description="工具名，必须是列出的之一")),
        arguments=(dict[str, Any], Field(default_factory=dict, description=_ARGUMENTS_DESCRIPTION)),
        __validators__={"_no_stringified_references":
                        field_validator("arguments")(_reject_stringified_references)},
    )
    return create_model(
        "TaskPlanConstrained",
        __config__=ConfigDict(extra="forbid"),
        tasks=(list[spec], Field(description="任务列表；无依赖的任务可并行")),
    )


class ResultReview(BaseModel):
    """对一次工具调用结果的校验判定。

    corrected_arguments 是「自动修正」的落点，但它只是**提议**：
    能不能派发由 `graph.verify.accept_correction` 确定性地决定。
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="被校验的任务 id")
    passed: bool = Field(description="结果是否达成了该任务描述要求的事")
    reason: str = Field(description="一句话说明判断依据")
    corrected_arguments: dict[str, Any] | None = Field(
        default=None,
        description="仅当失败原因是参数写错、且能从用户目标断定正确取值时填写："
                    "修正后的【完整】参数对象；无法断定就填 null，不要猜")


class ResultReviewBatch(BaseModel):
    """一轮执行里全部待校验结果的判定，一次调用给全。"""

    model_config = ConfigDict(extra="forbid")

    reviews: list[ResultReview] = Field(description="每个待校验任务一条判定")


class GoalReview(BaseModel):
    """对整轮执行状态的校验判定。"""

    model_config = ConfigDict(extra="forbid")

    achieved: bool = Field(description="用户目标是否已经达成")
    gap: str = Field(default="", description="未达成时说明还差什么，一句话；达成时留空")


def result_review_model_for(task_ids: Sequence[str]) -> type[BaseModel]:
    """按本轮待校验的任务收紧 schema：task_id 成为枚举。

    与工具名收进 enum 同一个道理 —— 判定挂错任务比判错更难察觉，
    在解析阶段拒掉，比事后去对齐 id 可靠。
    """
    if not task_ids:
        return ResultReviewBatch

    review = create_model(
        "ResultReviewConstrained",
        __config__=ConfigDict(extra="forbid"),
        task_id=(Literal[tuple(task_ids)],  # type: ignore[valid-type]
                 Field(description="被校验的任务 id，必须是列出的之一")),
        passed=(bool, Field(description="结果是否达成了该任务描述要求的事")),
        reason=(str, Field(description="一句话说明判断依据")),
        corrected_arguments=(
            dict[str, Any] | None,
            Field(default=None,
                  description="仅当失败原因是参数写错、且能从用户目标断定正确取值时填写："
                              "修正后的【完整】参数对象；无法断定就填 null，不要猜")),
    )
    return create_model(
        "ResultReviewBatchConstrained",
        __config__=ConfigDict(extra="forbid"),
        reviews=(list[review], Field(description="每个待校验任务一条判定")),
    )
