import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SecurityScanResult:
    allowed: bool
    risk_level: str
    matched_rules: tuple[str, ...]


class DocumentSecurityScanner:
    """Detect high-signal prompt injection instructions before indexing."""

    RULES = {
        "ignore_instructions": ( #匹配“忽略之前/以上/系统指令”类话术，意图让 AI 抛弃原有约束。
            r"ignore\s+(all\s+)?(previous|above|system)\s+instructions?",
            r"忽略.{0,12}(之前|以上|系统).{0,8}(指令|提示)",
        ),
        "system_prompt_extraction": (#匹配“把你的 system prompt 显示出来”这类套取系统配置的请求。
            r"(reveal|show|print|return).{0,20}system\s+prompt",
            r"(输出|显示|泄露|返回).{0,16}(系统提示词|系统指令)",
        ),
        "role_override": (  #匹配“你现在是管理员/开发者模式”类话术，试图让 AI 改变身份设定（如经典的 “DAN” 越狱就属于此类）。
            r"you\s+are\s+now.{0,30}(assistant|developer|system)",
            r"你现在是.{0,20}(助手|系统|管理员)",
        ),
        "secret_exfiltration": (#匹配“打印你的 API key / access token / 密码”。api[ _-]?key 表示 api key、api_key、apikey、api-key 都能命中。
            r"(reveal|print|return).{0,20}(api[ _-]?key|access[ _-]?token|secret)",
            r"(输出|显示|泄露).{0,16}(密钥|令牌|密码)",
        ),
        "command_execution": (#匹配“执行 shell 命令”类请求，防止 AI 被诱导执行危险操作。
            r"(execute|run).{0,12}(shell|terminal|system)\s+command",
            r"执行.{0,12}(终端|系统|shell).{0,8}命令",
        ),
    }

    def scan(self, text):
        matched = []
        for rule_name, patterns in self.RULES.items():
            if any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns):
                matched.append(rule_name)
        return SecurityScanResult(
            allowed=not matched,
            risk_level="high" if matched else "low",
            matched_rules=tuple(matched),
        )
