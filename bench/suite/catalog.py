"""评测用的系统工具目录。

与 miagent.mock_server.tools 的区别：那里的工具为演示而设，带有内建的故障
（第一家餐厅总订满、同步第一次总超时）；这里的工具**只做纯函数**——输出完全
由参数决定，同样的参数永远得到同样的结果。故障另由用例声明、在评测时注入。
这样数据集生成时就能直接算出每个调用的结果，下游引用该结果的参数取值也就
有了确定的标注答案。

每个工具除了声明（名称、描述、参数 schema、权限、内存开销）之外，还带两项
只供数据集使用的语义：

  produces   输出能被下游当作哪一类值引用（phone / address / place / text /
             when / order_id / song_id / city），None 表示输出只给人看
  accepts    哪些参数可以接收哪一类值的引用

输出能被引用的工具都返回「裸值」（一个号码、一个地址），不带说明文字：
引用传递的是整个结果（见 miagent.graph.dataflow），带说明文字的结果无法
直接作为下游参数。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from miagent.mock_server.tools import SystemTool

# 业务上存在的订单号形态。其余单号查无此单，服务端置 isError。
ORDER_ID = re.compile(r"^DD\d{10}$")


def pick(options: Sequence[str], *keys: Any) -> str:
    """按 keys 确定性地选一项。不用内置 hash：它按进程随机化。"""
    digest = hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def digits(n: int, *keys: Any) -> str:
    """按 keys 确定性地生成 n 位数字串。"""
    digest = hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()
    return str(int(digest[:16], 16))[-n:].rjust(n, "0")


_SLOTS = ("09:00", "10:30", "14:00", "15:30", "19:00", "20:30")
_PLACES = {
    "餐厅": ("云海肴(清河店)", "外婆家(五彩城店)", "西贝莜面村(华联店)"),
    "火锅店": ("海底捞(清河店)", "巴奴毛肚火锅(万象汇店)", "小龙坎(五彩城店)"),
    "咖啡馆": ("Manner Coffee(清河店)", "瑞幸咖啡(五彩城店)", "星巴克(万象汇店)"),
    "药店": ("同仁堂药店(清河店)", "国大药房(安宁庄店)", "老百姓大药房(小营店)"),
    "加油站": ("中石化(清河加油站)", "中石油(安宁庄加油站)", "壳牌(上地加油站)"),
    "超市": ("物美超市(清河店)", "盒马鲜生(五彩城店)", "永辉超市(华联店)"),
    "充电站": ("特来电(万象汇充电站)", "国家电网(清河充电站)", "星星充电(上地充电站)"),
    "书店": ("西西弗书店(五彩城店)", "中信书店(万象汇店)", "新华书店(清河店)"),
    "健身房": ("乐刻运动(清河店)", "超级猩猩(五彩城店)", "威尔士健身(华联店)"),
    "停车场": ("五彩城地下停车场", "万象汇P2停车场", "清河站东停车场"),
}
_STREETS = ("清河中街68号", "安宁庄东路18号", "上地十街10号", "小营西路33号",
            "西二旗大街39号", "毛纺路5号", "朱房路甲2号", "永泰庄北路1号")
_WEATHER = ("晴，22℃", "多云，18℃", "小雨，15℃", "阴，20℃", "晴转多云，25℃")
_UNREAD = ("「明早九点开会，别迟到」", "「快递已放驿站，取件码 3-2-1508」",
           "「周末聚餐改到周日中午」", "「报销单已审批通过」")
_CLIPBOARD = "明天上午十点在三号会议室开项目评审会"
_LOCATION = "北京市海淀区清河中街68号"
_CITY = "北京"
_LANG_TAG = {"英文": "EN", "日文": "JA", "韩文": "KO", "法文": "FR", "德文": "DE"}


def _note_content(title: str) -> str:
    return {
        "购物清单": "牛奶、鸡蛋、全麦面包、苹果",
        "会议纪要": "下周三前提交评审材料，周五上线",
        "读书笔记": "先写测试，再写实现",
        "旅行计划": "周六出发去杭州，住两晚",
        "待办事项": "交电费、取快递、预约体检",
        "健身计划": "周一三五跑步五公里",
    }.get(title, f"{title}：暂无内容")


# ---------------- 处理函数：全部是纯函数 ----------------


def _query_calendar(a: dict[str, Any]) -> str:
    return f"{a['when']} {pick(('19:00-22:00', '14:00-17:00', '全天'), 'cal', a['when'])} 空闲"


def _find_free_slot(a: dict[str, Any]) -> str:
    return f"{a['date']} {pick(_SLOTS, 'slot', a['date'])}"


def _recommend_place(a: dict[str, Any]) -> str:
    options = _PLACES.get(a["category"]) or (f"{a['category']}(清河店)",)
    return pick(options, "place", a["category"])


def _search_nearby(a: dict[str, Any]) -> str:
    options = _PLACES.get(a["category"]) or (f"{a['category']}(清河店)",)
    return f"附近{a['category']}：" + "、".join(options)


def _lookup_phone(a: dict[str, Any]) -> str:
    return "13" + digits(9, "phone", a["name"])


def _lookup_address(a: dict[str, Any]) -> str:
    return "北京市海淀区" + pick(_STREETS, "addr", a["name"])


def _latest_order(a: dict[str, Any]) -> str:
    return "DD" + digits(10, "order", a["platform"])


def _query_order(a: dict[str, Any]) -> str | None:
    if not ORDER_ID.match(a["order_id"]):
        return None
    return f"订单 {a['order_id']}：{pick(('已发货', '待发货', '已签收'), 'status', a['order_id'])}"


def _track_package(a: dict[str, Any]) -> str | None:
    if not ORDER_ID.match(a["order_id"]):
        return None
    return pick(("快递已到达海淀清河驿站，待取件", "快递正在派送中，预计今天送达",
                 "快递已从北京分拨中心发出"), "track", a["order_id"])


def _translate(a: dict[str, Any]) -> str:
    return f"[{_LANG_TAG.get(a['target_lang'], a['target_lang'])}] {a['text']}"


def _search_music(a: dict[str, Any]) -> str:
    return "song_" + digits(6, "song", a["keyword"])


def _book_restaurant(a: dict[str, Any]) -> str:
    people = f"，{a['people']} 人" if "people" in a else ""
    return f"已预订 {a['name']}，{a['when']}{people}"


def _navigate(a: dict[str, Any]) -> str:
    mode = a.get("mode", "驾车")
    return f"已开始{mode}导航至 {a['destination']}，预计 {10 + int(digits(2, 'eta', a['destination'], mode)) % 40} 分钟"


def _call_taxi(a: dict[str, Any]) -> str:
    return f"已呼叫快车前往 {a['destination']}，司机约 {2 + int(digits(1, 'taxi', a['destination']))} 分钟后到达"


def _query_traffic(a: dict[str, Any]) -> str:
    return f"前往 {a['destination']}：{pick(('畅通', '缓行', '拥堵'), 'traffic', a['destination'])}"


def _send_email(a: dict[str, Any]) -> str:
    return f"已向 {a['to']} 发送邮件「{a['subject']}」"


@dataclass(frozen=True)
class Spec:
    """一个工具及其供数据集使用的语义。"""

    tool: SystemTool
    produces: str | None = None
    accepts: dict[str, str] = field(default_factory=dict)
    side_effect: bool = False

    @property
    def name(self) -> str:
        return self.tool.name

    def run(self, args: dict[str, Any]) -> str | None:
        return self.tool.handler(args)


def _schema(required: dict[str, tuple[str, str]],
            optional: dict[str, tuple[str, str]] | None = None) -> dict[str, Any]:
    """参数 {名: (JSON 类型, 说明)}。"""
    props = {**required, **(optional or {})}
    return {"type": "object",
            "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()},
            "required": list(required)}


def _spec(name: str, description: str, schema: dict[str, Any], permission: str | None,
          memory_mb: int, handler: Callable[[dict[str, Any]], str | None], *,
          produces: str | None = None, accepts: dict[str, str] | None = None,
          side_effect: bool = False) -> Spec:
    return Spec(SystemTool(name, description, schema, permission, memory_mb, handler),
                produces=produces, accepts=accepts or {}, side_effect=side_effect)


S = "string"
SPECS: list[Spec] = [
    # ---- 日历与提醒 ----
    _spec("system.query_calendar", "查询指定时段日历是否空闲",
          _schema({"when": (S, "时段，如 今晚、明天下午")}), "calendar.read", 6,
          _query_calendar),
    _spec("system.find_free_slot", "在指定日期的日历里找一个空闲时间点",
          _schema({"date": (S, "日期，如 明天、周六")}), "calendar.read", 6,
          _find_free_slot, produces="when"),
    _spec("system.create_event", "在日历中创建日程",
          _schema({"title": (S, "日程标题"), "when": (S, "时间，如 明天 19:00")}),
          "calendar.write", 8, lambda a: f"已创建日程「{a['title']}」于 {a['when']}",
          accepts={"when": "when"}, side_effect=True),
    _spec("system.create_reminder", "创建一条提醒",
          _schema({"content": (S, "提醒内容"), "when": (S, "提醒时间")}),
          "reminder.write", 6, lambda a: f"已设置提醒：{a['when']} {a['content']}",
          accepts={"content": "text", "when": "when"}, side_effect=True),
    _spec("system.set_alarm", "创建一个闹钟",
          _schema({"time": (S, "闹钟时间，24 小时制 HH:MM")}, {"label": (S, "备注")}),
          "alarm.write", 8,
          lambda a: f"已创建 {a['time']} 的闹钟（{a.get('label', '无备注')}）",
          side_effect=True),

    # ---- 位置与出行 ----
    _spec("system.get_location", "获取设备当前所在地址", _schema({}),
          "location.coarse", 6, lambda a: _LOCATION),
    _spec("system.get_city", "获取设备当前所在城市", _schema({}),
          "location.coarse", 4, lambda a: _CITY, produces="city"),
    _spec("system.search_nearby", "搜索附近某类地点，返回候选列表",
          _schema({"category": (S, "类别，如 餐厅、药店")}), "location.fine", 14,
          _search_nearby, produces="text"),
    _spec("system.recommend_place", "推荐附近一家某类地点，返回店名",
          _schema({"category": (S, "类别，如 餐厅、咖啡馆")}), "location.fine", 14,
          _recommend_place, produces="place"),
    _spec("system.navigate", "开始导航到目的地",
          _schema({"destination": (S, "目的地名称或地址")}, {"mode": (S, "出行方式：驾车、步行、骑行、公交")}),
          "location.fine", 24, _navigate,
          accepts={"destination": "address|place"}, side_effect=True),
    _spec("system.call_taxi", "呼叫网约车前往目的地",
          _schema({"destination": (S, "目的地名称或地址")}), "location.fine", 16,
          _call_taxi, accepts={"destination": "address|place"}, side_effect=True),
    _spec("system.query_traffic", "查询前往目的地的路况",
          _schema({"destination": (S, "目的地名称或地址")}), "location.fine", 12,
          _query_traffic, produces="text", accepts={"destination": "address|place"}),
    _spec("system.book_restaurant", "预订餐厅",
          _schema({"name": (S, "餐厅名"), "when": (S, "用餐时间")},
                  {"people": ("integer", "用餐人数")}),
          "booking.write", 8, _book_restaurant, produces="text",
          accepts={"name": "place", "when": "when"}, side_effect=True),

    # ---- 通讯 ----
    _spec("system.lookup_phone", "按姓名查询联系人的手机号",
          _schema({"name": (S, "联系人姓名")}), "contacts.read", 10,
          _lookup_phone, produces="phone"),
    _spec("system.lookup_address", "按姓名查询联系人的住址",
          _schema({"name": (S, "联系人姓名")}), "contacts.read", 10,
          _lookup_address, produces="address"),
    _spec("system.send_sms", "发送短信",
          _schema({"to": (S, "收件人手机号"), "text": (S, "短信正文")}), "sms.send", 12,
          lambda a: f"已向 {a['to']} 发送短信",
          accepts={"to": "phone", "text": "text"}, side_effect=True),
    _spec("system.make_call", "拨打电话",
          _schema({"number": (S, "电话号码")}), "phone.call", 10,
          lambda a: f"正在呼叫 {a['number']}", accepts={"number": "phone"}, side_effect=True),
    _spec("system.send_email", "发送邮件",
          _schema({"to": (S, "收件人邮箱"), "subject": (S, "邮件标题"), "body": (S, "邮件正文")}),
          "email.send", 14, _send_email, accepts={"body": "text"}, side_effect=True),
    _spec("system.check_unread", "读取某个应用的最新一条未读消息",
          _schema({"app": (S, "应用名，如 微信、钉钉")}), "notification.read", 8,
          lambda a: pick(_UNREAD, "unread", a["app"]), produces="text"),

    # ---- 内容 ----
    _spec("system.read_clipboard", "读取剪贴板里的文字", _schema({}),
          "clipboard.read", 4, lambda a: _CLIPBOARD, produces="text"),
    _spec("system.read_note", "按标题读取一条笔记的内容",
          _schema({"title": (S, "笔记标题")}), "notes.read", 8,
          lambda a: _note_content(a["title"]), produces="text"),
    _spec("system.create_note", "新建一条笔记",
          _schema({"title": (S, "笔记标题"), "content": (S, "笔记内容")}), "notes.write", 8,
          lambda a: f"已新建笔记「{a['title']}」", accepts={"content": "text"}, side_effect=True),
    _spec("system.translate", "把一段文字翻译成目标语言",
          _schema({"text": (S, "要翻译的文字"), "target_lang": (S, "目标语言，如 英文、日文")}),
          None, 30, _translate, produces="text", accepts={"text": "text"}),
    _spec("system.query_weather", "查询某个城市某天的天气",
          _schema({"city": (S, "城市"), "date": (S, "日期，如 今天、明天")}), None, 5,
          lambda a: f"{a['city']}{a['date']}{pick(_WEATHER, 'wx', a['city'], a['date'])}",
          produces="text", accepts={"city": "city"}),

    # ---- 订单与快递 ----
    _spec("system.latest_order", "查询某个购物平台上最近一笔订单的订单号",
          _schema({"platform": (S, "平台，如 淘宝、京东")}), "orders.read", 8,
          _latest_order, produces="order_id"),
    _spec("system.query_order", "按订单号查询订单状态",
          _schema({"order_id": (S, "订单号")}), "orders.read", 6,
          _query_order, produces="text", accepts={"order_id": "order_id"}),
    _spec("system.track_package", "按订单号查询物流进度",
          _schema({"order_id": (S, "订单号")}), "orders.read", 6,
          _track_package, produces="text", accepts={"order_id": "order_id"}),

    # ---- 媒体 ----
    _spec("system.search_music", "按关键词搜索歌曲，返回歌曲 id",
          _schema({"keyword": (S, "歌名、歌手或风格")}), "media.read", 12,
          _search_music, produces="song_id"),
    _spec("system.play_music", "播放指定歌曲",
          _schema({"song_id": (S, "歌曲 id")}), "media.play", 20,
          lambda a: f"正在播放 {a['song_id']}", accepts={"song_id": "song_id"},
          side_effect=True),

    # ---- 设备 ----
    _spec("system.get_battery", "查询设备当前电量与充电状态", _schema({}), None, 4,
          lambda a: "电量 63%，未在充电", produces="text"),
    _spec("system.set_volume", "设置媒体音量",
          _schema({"level": ("integer", "音量，0 到 100")}), "settings.write", 4,
          lambda a: f"音量已设为 {a['level']}", side_effect=True),
    _spec("system.set_brightness", "设置屏幕亮度",
          _schema({"level": ("integer", "亮度，0 到 100")}), "settings.write", 4,
          lambda a: f"亮度已设为 {a['level']}", side_effect=True),
    _spec("system.toggle_wifi", "打开或关闭 WLAN",
          _schema({"on": ("boolean", "true 为打开，false 为关闭")}), "settings.write", 4,
          lambda a: f"WLAN 已{'打开' if a['on'] else '关闭'}", side_effect=True),
    _spec("system.toggle_dnd", "打开或关闭勿扰模式",
          _schema({"on": ("boolean", "true 为打开，false 为关闭")}), "settings.write", 4,
          lambda a: f"勿扰模式已{'打开' if a['on'] else '关闭'}", side_effect=True),
    _spec("system.sync_settings", "把本机设置同步到账号", _schema({}), None, 10,
          lambda a: "设置已同步至云端", side_effect=True),
    _spec("system.capture_screen", "截取当前屏幕并保存到相册", _schema({}),
          "screen.capture", 200, lambda a: "截图已保存到相册", side_effect=True),
    _spec("system.record_screen", "录制屏幕若干秒并保存到相册",
          _schema({"seconds": ("integer", "录制时长，秒")}), "screen.capture", 300,
          lambda a: f"已录制 {a['seconds']} 秒屏幕", side_effect=True),
]

CATALOG: dict[str, Spec] = {s.name: s for s in SPECS}

# 全部权限。用例默认全部授予，权限类用例从中去掉若干项。
ALL_PERMISSIONS: tuple[str, ...] = tuple(sorted(
    {s.tool.required_permission for s in SPECS if s.tool.required_permission}))

# 评测默认的端侧配额。内存上限取协议默认值（ResourceBudget 的占位值，无外部依据）；
# 超过它的工具（录屏）在默认配额下必然触发 MC-4001。
# 能订座的地点类别：推荐结果要交给订餐厅时，类别只能取这几种
DINING = ("餐厅", "火锅店", "咖啡馆")

DEFAULT_BUDGET = {"max_memory_mb": 256, "max_concurrent_calls": 2,
                  "max_call_timeout_ms": 5000, "power_saving": False}


def accepts(spec: Spec, param: str, value_type: str) -> bool:
    return value_type in spec.accepts.get(param, "").split("|")
