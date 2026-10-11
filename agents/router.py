from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    domain: str
    category: str
    question_type: str
    confidence: float
    matched_keywords: tuple[str, ...]


class DomainRouterAgent:
    """Routes a question without spending an extra model call."""

    DOMAIN_RULES = {
        "hr": ("HR制度", ("请假", "年假", "病假", "考勤", "补卡", "调休", "离职", "加班", "社保", "公积金", "劳动合同", "调休")),
        "finance": ("财务制度", ("报销", "差旅", "发票", "借款", "费用", "付款", "预算", "税务", "薪酬", "福利", "冲销")),
        "it": ("IT支持", ("vpn", "密码", "账号", "权限", "软件", "电脑", "网络", "邮箱", "系统", "蓝屏", "工单", "故障", "安装", "设备", "领用", "重置")),
        "admin": ("行政流程", ("采购", "办公用品", "会议室", "用印", "合同", "资产", "车辆", "接待", "比价", "报价", "下单")),
        "employee": ("员工手册", ("员工手册", "行为规范", "试用期", "转正", "福利", "工作时间", "上班", "迟到", "签到", "午休", "着装", "纪律", "工作制")),
        "general": (
            "综合公告",
            ("通知", "公告", "总结", "概括", "重点", "新员工", "入职", "权限开通", "开通", "办理"),
        ),
    }

    def route(self, question, selected_category="全部"):
        question_text = question.strip().lower()
        question_type = self._infer_question_type(question_text)

        if selected_category != "全部":
            return RouteDecision(
                domain=self._domain_for_category(selected_category),
                category=selected_category,
                question_type=question_type,
                confidence=1.0,
                matched_keywords=(),
            )

        candidates = []
        for domain, (category, keywords) in self.DOMAIN_RULES.items():
            matched = tuple(keyword for keyword in keywords if keyword in question_text)
            if matched:
                # 优先级：匹配关键词总长度越长，优先级越高（更具体的匹配）
                keyword_total_len = sum(len(k) for k in matched)
                candidates.append((len(matched), keyword_total_len, domain, category, matched))

        if not candidates:
            return RouteDecision("general", "全部", question_type, 0.0, ())

        # 按匹配数量降序，数量相同时按关键词总长度降序
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        match_count, _, domain, category, matched = candidates[0]

        confidence = min(0.55 + 0.15 * match_count, 0.95)
        return RouteDecision(domain, category, question_type, confidence, matched)

    def _infer_question_type(self, question):
        if any(keyword in question for keyword in ("材料", "附件", "提交什么", "要带什么", "需要什么")):
            return "material_list"
        if any(
            keyword in question for keyword in ("流程", "步骤", "怎么走", "怎么发起", "如何办理", "怎么办")
        ):
            return "process_guide"
        if any(keyword in question for keyword in ("总结", "概括", "通知重点", "提炼")):
            return "notice_summary"
        return "policy_qa"

    def _domain_for_category(self, category):
        for domain, (configured_category, _) in self.DOMAIN_RULES.items():
            if configured_category == category:
                return domain
        return "general"
