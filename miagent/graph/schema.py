"""计划的结构化 schema。

★ 这里是「计划长什么样」的唯一事实来源：

    Pydantic 模型  ──model_json_schema()──→  喂给模型的 schema
                   ──model_validate()────→  校验模型的输出

  写在 prompt 里的格式说明与实际校验逻辑是两份拷贝，改一处忘另一处
  就会出现「提示词说要 steps、校验器却认 plan」这种问题。由模型定义
  自动生成 schema 之后，两者不可能不一致。
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, create_model


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")   # 多余字段直接判不合格

    tool: str = Field(description="要调用的工具名")
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具参数")
    reason: str = Field(default="", description="为什么需要这一步")


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[PlanStep] = Field(description="按顺序执行的步骤；无需工具时为空数组")


def plan_model_for(tool_names: Sequence[str]) -> type[BaseModel]:
    """按当前可用工具生成一个收紧的 Plan 模型。

    把工具名做成枚举写进 schema，带来两个收益：
      · 模型看到的 schema 里直接列出了合法工具名，减少幻觉
      · 万一还是幻觉了，在【解析阶段】就被拒（AG-1001），
        而不是白跑一步再报 AG-2001 —— 省一次工具调用

    这也是将来接约束解码时的落点：同一份 schema 可以直接转成
    采样语法，让不合法的工具名在物理上无法被生成。
    """
    if not tool_names:
        return Plan

    tool_field = Literal[tuple(tool_names)]  # type: ignore[valid-type]
    step_model = create_model(
        "PlanStepConstrained",
        __config__=ConfigDict(extra="forbid"),
        tool=(tool_field, Field(description="要调用的工具名，必须是列出的之一")),
        arguments=(dict[str, Any], Field(default_factory=dict, description="工具参数")),
        reason=(str, Field(default="", description="为什么需要这一步")),
    )
    return create_model(
        "PlanConstrained",
        __config__=ConfigDict(extra="forbid"),
        steps=(list[step_model], Field(description="按顺序执行的步骤；无需工具时为空数组")),
    )
