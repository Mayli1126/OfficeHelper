import unittest

from agents.router import DomainRouterAgent


class DomainRouterAgentTests(unittest.TestCase):
    def setUp(self):
        self.router = DomainRouterAgent()

    def test_routes_finance_material_question(self):
        route = self.router.route("差旅报销需要提交什么材料？")
        self.assertEqual(route.domain, "finance")
        self.assertEqual(route.category, "财务制度")
        self.assertEqual(route.question_type, "material_list")
        self.assertGreaterEqual(route.confidence, 0.7)

    def test_explicit_category_has_priority(self):
        route = self.router.route("密码忘了怎么办？", selected_category="HR制度")
        self.assertEqual(route.category, "HR制度")
        self.assertEqual(route.confidence, 1.0)

    def test_unknown_question_keeps_global_search(self):
        route = self.router.route("公司的核心价值观是什么？")
        self.assertEqual(route.category, "全部")
        self.assertEqual(route.confidence, 0.0)

    def test_routes_onboarding_permission_to_general_documents(self):
        route = self.router.route("新员工如何完成权限开通？")
        self.assertEqual(route.category, "综合公告")


if __name__ == "__main__":
    unittest.main()
