"""用户话术：参数取值的采样与每个工具的说法模板。

参数取值分两层：value 是标注答案（工具该收到的值），surface 是用户嘴里的
说法。多数参数两者相同 —— 规划器的规则是「用户说出的内容原样填入参数」；
不同的只有工具声明了格式的参数（闹钟的 HH:MM、音量的整数、开关的布尔），
这时 value 是按 schema 换算后的值，surface 是口语说法，accept 列出评分时
也算对的写法。

说法模板按「哪些参数是引用」分组：同一个工具，参数由用户直接说出与取自
上一步的结果，说法不同（「给 138… 发短信」与「把译文发短信给他」）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from .catalog import digits

# ---------------- 词表 ----------------

NAMES = ("张伟", "王芳", "李娜", "刘洋", "陈静", "杨帆", "赵磊", "黄敏", "周杰", "吴婷",
         "徐明", "孙丽", "马超", "朱琳", "胡军", "郭晓", "何平", "林峰", "高远", "罗佳",
         "梁宇", "宋雪", "郑浩", "谢楠", "韩梅", "唐亮", "冯雨", "董洁", "萧然", "程诚",
         "妈妈", "爸爸", "老婆", "老公", "姐姐", "哥哥", "王老师", "李经理", "刘医生", "陈总")
DAYS = ("今天", "明天", "后天", "周六", "周日", "下周一", "下周三", "下周五")
PERIODS = ("今晚", "明天上午", "明天下午", "明晚", "后天上午", "周六全天", "周日下午", "下周一上午")
TITLES = ("晚餐", "周会", "看牙医", "健身", "家长会", "项目评审", "生日聚会", "体检",
          "面试", "读书会", "部门团建", "客户拜访", "英语课", "瑜伽课", "年度述职")
REMINDERS = ("交电费", "取快递", "给妈妈打电话", "吃药", "还信用卡", "浇花", "带伞",
             "交房租", "续费会员", "倒垃圾", "预约体检", "给车加油")
CATEGORIES = ("餐厅", "火锅店", "咖啡馆", "药店", "加油站", "超市", "充电站", "书店", "健身房", "停车场")
DESTINATIONS = ("北京南站", "首都机场T3航站楼", "国家图书馆", "五道口购物中心", "颐和园",
                "北京西站", "中关村软件园", "望京SOHO", "奥林匹克森林公园", "798艺术区",
                "朝阳大悦城", "北京大学东门", "协和医院", "西单大悦城", "三里屯太古里")
MODES = ("驾车", "步行", "骑行", "公交")
SMS = ("我晚点到", "会议改到明天上午", "今晚不回家吃饭", "到家了", "帮我带杯咖啡",
       "快递帮我收一下", "路上堵车，晚二十分钟", "周末一起吃饭吧", "文件已经发你邮箱了",
       "记得带钥匙", "生日快乐", "明天早上八点楼下见")
EMAIL_USERS = ("zhangwei", "wangfang", "hr", "team", "lina.li", "boss", "support", "chenjing")
SUBJECTS = ("周报", "会议安排", "请假申请", "项目进度", "报销材料", "面试反馈", "出差行程")
BODIES = ("附件是本周的工作总结，请查收", "明天下午三点开会，请准时参加", "我下周一请假一天",
          "项目已完成七成，预计月底交付", "发票已经整理好了", "候选人表现不错，建议进入下一轮")
APPS = ("微信", "钉钉", "企业微信", "飞书", "短信", "邮件")
NOTE_TITLES = ("购物清单", "会议纪要", "读书笔记", "旅行计划", "待办事项", "健身计划")
NOTE_CONTENTS = ("周五前把合同寄出", "买一束花", "下周要交的报告还差两页", "护照记得续签",
                 "每天喝八杯水", "周末去看电影")
TEXTS = ("明天见", "谢谢你的帮助", "会议推迟半小时", "我在路上了", "请把文件发给我",
         "今天天气很好", "祝你生日快乐", "这个方案我同意")
LANGS = ("英文", "日文", "韩文", "法文", "德文")
CITIES = ("北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "重庆", "厦门", "青岛")
PLATFORMS = ("淘宝", "京东", "拼多多", "美团", "唯品会")
KEYWORDS = ("周杰伦", "陈奕迅", "轻音乐", "古典钢琴", "摇滚", "邓紫棋", "爵士", "五月天", "白噪音", "粤语老歌")
RESTAURANTS = ("小馆 A", "小馆 B", "鼎泰丰", "大董", "四季民福", "局气", "眉州东坡",
               "南京大牌档", "新荣记", "胡大饭馆", "海底捞", "外婆家")
ALARM_LABELS = ("起床", "开会", "吃药", "健身", "接孩子")

_CN_NUM = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八",
           9: "九", 10: "十", 11: "十一", 12: "十二"}


@dataclass(frozen=True)
class Value:
    """一个参数的取值。value 是标注答案，surface 是用户的说法。

    referred 为真表示用户没有说出这个值，而是指代了上文的结论（「刚才查到的号码」）：
    标注答案是上文的值，说法按引用的形式组织。"""

    value: Any
    surface: str
    accept: tuple[Any, ...] = field(default_factory=tuple)
    referred: bool = False


def _same(options: tuple[str, ...]) -> Callable[[random.Random], Value]:
    return lambda rng: (lambda v: Value(v, v))(rng.choice(options))


def _quoted(options: tuple[str, ...]) -> Callable[[random.Random], Value]:
    """自由文本：用户用引号框出原文，标注答案是引号里的内容。"""
    return lambda rng: (lambda v: Value(v, f"「{v}」"))(rng.choice(options))


def _when(rng: random.Random) -> Value:
    """日程类时间：原样填入，用户怎么说就怎么填。"""
    day = rng.choice(DAYS)
    hour = rng.choice((9, 10, 14, 15, 18, 19, 20))
    minute = rng.choice((0, 30))
    v = f"{day} {hour:02d}:{minute:02d}"
    return Value(v, v)


def _alarm_time(rng: random.Random) -> Value:
    """闹钟时间：schema 要求 HH:MM，口语说法要换算。"""
    hour = rng.choice((5, 6, 7, 8, 9, 13, 14, 21, 22))
    minute = rng.choice((0, 15, 30, 45))
    v = f"{hour:02d}:{minute:02d}"
    shown = hour if hour <= 12 else hour - 12
    part = "早上" if hour < 12 else ("下午" if hour < 18 else "晚上")
    tail = {0: "点", 15: "点一刻", 30: "点半", 45: "点四十五"}[minute]
    spoken = rng.choice((v, f"{part}{_CN_NUM[shown]}{tail}", f"{part} {shown}{tail}"))
    return Value(v, spoken, (f"{hour}:{minute:02d}",))


def _level(rng: random.Random) -> Value:
    n = rng.choice(range(0, 101, 5))
    return Value(n, rng.choice((f"{n}", f"{n}%")), (str(n),))


def _switch(rng: random.Random) -> Value:
    on = rng.random() < 0.5
    return Value(on, rng.choice(("打开", "开一下", "开启")) if on else rng.choice(("关掉", "关闭", "关一下")))


def _seconds(rng: random.Random) -> Value:
    n = rng.choice((10, 15, 30, 60, 120))
    spoken = {60: "一分钟", 120: "两分钟"}.get(n, f"{n} 秒")
    return Value(n, spoken, (str(n),))


def _people(rng: random.Random) -> Value:
    n = rng.choice(range(2, 9))
    return Value(n, rng.choice((f"{n} 个人", f"{_CN_NUM[n]}个人", f"{_CN_NUM[n]}位")), (str(n),))


def _phone(rng: random.Random) -> Value:
    v = "1" + rng.choice("3589") + digits(9, "lit", rng.random())
    return Value(v, v)


def _email(rng: random.Random) -> Value:
    v = f"{rng.choice(EMAIL_USERS)}@example.com"
    return Value(v, v)


def _order_id(rng: random.Random) -> Value:
    v = "DD" + digits(10, "lit-order", rng.random())
    return Value(v, v)


# 参数取值的采样器：(工具, 参数) → 采样函数。缺省按参数名找。
BY_PARAM: dict[str, Callable[[random.Random], Value]] = {
    "when": _when, "date": _same(DAYS), "title": _same(TITLES), "content": _quoted(REMINDERS),
    "time": _alarm_time, "label": _same(ALARM_LABELS), "category": _same(CATEGORIES),
    "destination": _same(DESTINATIONS), "mode": _same(MODES), "name": _same(NAMES),
    "to": _phone, "text": _quoted(SMS), "number": _phone, "subject": _same(SUBJECTS),
    "body": _quoted(BODIES), "app": _same(APPS), "target_lang": _same(LANGS),
    "city": _same(CITIES), "platform": _same(PLATFORMS), "order_id": _order_id,
    "keyword": _same(KEYWORDS), "level": _level, "on": _switch, "seconds": _seconds,
    "people": _people,
}
BY_TOOL_PARAM: dict[tuple[str, str], Callable[[random.Random], Value]] = {
    ("system.query_calendar", "when"): _same(PERIODS),
    ("system.book_restaurant", "name"): _same(RESTAURANTS),
    ("system.send_email", "to"): _email,
    ("system.create_note", "title"): _same(NOTE_TITLES),
    ("system.create_note", "content"): _quoted(NOTE_CONTENTS),
    ("system.read_note", "title"): _same(NOTE_TITLES),
    ("system.translate", "text"): _quoted(TEXTS),
}

# 用户不会直接说出的参数：只能取自上一步的结果。
REF_ONLY = {("system.play_music", "song_id")}


def sample(tool: str, param: str, rng: random.Random) -> Value:
    return (BY_TOOL_PARAM.get((tool, param)) or BY_PARAM[param])(rng)


# ---------------- 说法模板 ----------------
#
# 键是引用参数的集合；模板里 {参数} 处填用户的说法或上一步结果的指代。
# 可选参数出现与否由生成器决定，模板里可选参数写成 {?参数:前缀…后缀}，
# 参数缺席时整段省略。

T = dict[frozenset[str], tuple[str, ...]]


def _t(**groups: tuple[str, ...]) -> T:
    """plain=无引用；其余键名是用下划线连接的引用参数。"""
    return {frozenset() if k == "plain" else frozenset(k.split("__")): v for k, v in groups.items()}


PHRASES: dict[str, T] = {
    "system.query_calendar": _t(plain=("看看{when}有没有空", "查一下我{when}是否有空", "{when}日程空不空")),
    "system.find_free_slot": _t(plain=("在{date}找个空闲时间", "看看{date}什么时候有空档", "帮我找{date}的一个空闲时段")),
    "system.create_event": _t(
        plain=("在{when}建一个「{title}」的日程", "把{title}安排在{when}", "日历里加一个{when}的{title}"),
        when=("在{when}建一个「{title}」的日程", "把{title}安排在{when}")),
    "system.create_reminder": _t(
        plain=("{when}提醒我{content}", "设个提醒，{when}{content}", "{when}记得提醒我{content}"),
        when=("在{when}提醒我{content}", "到{when}提醒我{content}"),
        content=("{when}把{content}设成提醒", "{when}用{content}提醒我"),
        content__when=("在{when}用{content}提醒我",)),
    "system.set_alarm": _t(plain=("定一个{time}的闹钟{?label:，备注「…」}", "{time}叫我{?label:，标记为「…」}", "设个{time}的闹钟{?label:，写上「…」}")),
    "system.get_location": _t(plain=("看看我现在在哪", "获取一下我当前的位置", "定位一下我现在的地址")),
    "system.get_city": _t(plain=("看看我现在在哪个城市", "确认一下我所在的城市")),
    "system.search_nearby": _t(plain=("搜一下附近的{category}", "附近有哪些{category}", "找找周边的{category}")),
    "system.recommend_place": _t(plain=("推荐一家附近的{category}", "帮我挑一家附近的{category}", "找一家近点的{category}")),
    "system.navigate": _t(
        plain=("{?mode:…}导航去{destination}", "带我去{destination}{?mode:，…}", "开导航到{destination}{?mode:，…过去}"),
        destination=("{?mode:…}导航去{destination}", "再导航到{destination}{?mode:，…}", "带我去{destination}{?mode:，…}")),
    "system.call_taxi": _t(
        plain=("叫辆车去{destination}", "帮我打车到{destination}", "约个车去{destination}"),
        destination=("叫辆车去{destination}", "打车过去{destination}")),
    "system.query_traffic": _t(
        plain=("查查去{destination}的路况", "去{destination}堵不堵", "看下到{destination}的交通情况"),
        destination=("查查去{destination}的路况", "看看去{destination}堵不堵")),
    "system.book_restaurant": _t(
        plain=("订{name}，{when}{?people:，…}", "帮我在{name}订个位子，{when}{?people:，…}", "预订{when}的{name}{?people:，…}"),
        name=("订{name}，{when}{?people:，…}", "在{name}订{when}的位子{?people:，…}"),
        when=("在{when}订{name}{?people:，…}", "订{name}，时间就用{when}{?people:，…}"),
        name__when=("在{when}订{name}{?people:，…}",)),
    "system.lookup_phone": _t(plain=("查一下{name}的手机号", "找到{name}的电话", "看看{name}的号码是多少")),
    "system.lookup_address": _t(plain=("查一下{name}的住址", "找到{name}家的地址")),
    "system.send_sms": _t(
        plain=("给{to}发短信说{text}", "发条短信给{to}：{text}", "短信告诉{to}{text}"),
        to=("给{to}发短信说{text}", "发条短信给{to}：{text}"),
        text=("把{text}用短信发给{to}", "把{text}发短信给{to}"),
        text__to=("把{text}用短信发到{to}", "把{text}短信发给{to}")),
    "system.make_call": _t(plain=("给{number}打个电话", "拨打{number}"), number=("给{number}打个电话", "拨通{number}")),
    "system.send_email": _t(
        plain=("给{to}发封邮件，标题「{subject}」，正文{body}", "发邮件到{to}，主题{subject}，内容是{body}"),
        body=("把{body}发邮件给{to}，标题「{subject}」", "用邮件把{body}发到{to}，主题写{subject}")),
    "system.check_unread": _t(plain=("看看{app}的最新未读消息", "读一下{app}里最新的一条未读")),
    "system.read_clipboard": _t(plain=("读一下剪贴板", "看看剪贴板里是什么")),
    "system.read_note": _t(plain=("打开「{title}」这条笔记", "读一下笔记「{title}」")),
    "system.create_note": _t(
        plain=("新建一条笔记「{title}」，内容是{content}", "记一条笔记，标题{title}，写上{content}"),
        content=("把{content}存成笔记「{title}」", "新建笔记「{title}」，内容用{content}")),
    "system.translate": _t(
        plain=("把{text}翻译成{target_lang}", "{text}用{target_lang}怎么说"),
        text=("把{text}翻译成{target_lang}", "将{text}译成{target_lang}")),
    "system.query_weather": _t(
        plain=("{city}{date}天气怎么样", "查一下{city}{date}的天气"),
        city=("查一下{city}{date}的天气", "看看{city}{date}天气如何")),
    "system.latest_order": _t(plain=("查一下我在{platform}最近的一笔订单", "看看{platform}上最新的订单号")),
    "system.query_order": _t(
        plain=("查一下订单 {order_id} 的状态", "订单 {order_id} 到哪一步了"),
        order_id=("查一下{order_id}的状态", "看看{order_id}现在什么状态")),
    "system.track_package": _t(
        plain=("查一下订单 {order_id} 的物流", "订单 {order_id} 的快递到哪了"),
        order_id=("查一下{order_id}的物流", "看看{order_id}的快递到哪了")),
    "system.search_music": _t(plain=("搜一下{keyword}的歌", "找找{keyword}")),
    "system.play_music": _t(song_id=("播放{song_id}", "放一下{song_id}")),
    "system.get_battery": _t(plain=("看看还剩多少电", "查一下电量")),
    "system.set_volume": _t(plain=("音量调到 {level}", "把音量设成 {level}")),
    "system.set_brightness": _t(plain=("亮度调到 {level}", "屏幕亮度设为 {level}")),
    "system.toggle_wifi": _t(plain=("{on} WiFi", "把 WLAN {on}")),
    "system.toggle_dnd": _t(plain=("{on}勿扰模式", "把免打扰{on}")),
    "system.sync_settings": _t(plain=("把手机设置同步到账号", "同步一下本机设置")),
    "system.capture_screen": _t(plain=("截个屏", "截一下当前屏幕")),
    "system.record_screen": _t(plain=("录{seconds}屏幕", "录屏{seconds}")),
}

# 结果的指代：下游在同一句话里提到上一步的结果时怎么说。{参数} 取上一步的参数。
REFERENCES: dict[str, tuple[str, ...]] = {
    "system.query_calendar": ("日历的查询结果",),
    "system.find_free_slot": ("找到的那个时间", "那个空档"),
    "system.recommend_place": ("那家店", "推荐的那家"),
    "system.search_nearby": ("搜到的列表",),
    "system.get_location": ("我现在的位置",),
    "system.get_city": ("我所在的城市", "这个城市"),
    "system.lookup_phone": ("{name}", "这个号码", "对方"),
    "system.lookup_address": ("{name}家", "这个地址"),
    "system.check_unread": ("这条消息", "消息内容"),
    "system.read_clipboard": ("剪贴板里的内容", "这段文字"),
    "system.read_note": ("笔记内容", "这条笔记的内容"),
    "system.translate": ("翻译结果", "译文"),
    "system.query_weather": ("天气情况", "天气预报"),
    "system.latest_order": ("这笔订单", "这个订单"),
    "system.query_order": ("订单状态",),
    "system.track_package": ("物流信息", "快递进度"),
    "system.search_music": ("搜到的歌", "这首歌"),
    "system.query_traffic": ("路况",),
    "system.book_restaurant": ("预订信息", "订座结果"),
    "system.get_battery": ("电量情况",),
}

# 跨轮指代：上一轮的结论在这一轮被提起时怎么说。
PRIOR = {
    "phone": ("刚才查到的号码", "那个号码"),
    "place": ("刚才推荐的那家", "那家店"),
    "when": ("刚才找到的时间", "那个时间"),
    "order_id": ("刚才那笔订单", "那个订单"),
    "address": ("刚才那个地址", "那个地方"),
    "city": ("那个城市",),
    "song_id": ("刚才搜到的歌", "那首歌"),
    "text": ("刚才的内容", "上面那段"),
}

PREFIXES = ("", "", "", "帮我", "麻烦", "请", "小爱，", "嗯，")
SUFFIXES = ("", "", "", "，谢谢", "。", "，快点")
JOIN_PARALLEL = ("，", "，同时", "；另外", "，顺便", "，还有")
JOIN_SEQUENTIAL = ("，然后", "，再", "，接着", "，之后")
JOIN_AFTER_ALL = ("，这些都办完之后", "，以上都完成后", "，等前面都好了")
