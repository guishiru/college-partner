import unittest

from modules.employees import (
    LAYOUT_ANALYSIS_SPLIT,
    LAYOUT_CONVERSATION,
    LAYOUT_DOCUMENT_SPLIT,
    LAYOUTS,
    EmployeeError,
    can_invoke,
    get_employee,
    list_employees,
    require_available,
)


class EmployeeRegistryTests(unittest.TestCase):
    def test_every_employee_declares_a_known_layout(self):
        for employee in list_employees():
            with self.subTest(employee=employee.employee_id):
                self.assertIn(employee.layout, LAYOUTS)

    def test_layout_differs_by_employee(self):
        """布局是员工的属性。全都一样的话，这个字段就没有存在的意义。"""

        layouts = {employee.layout for employee in list_employees()}
        self.assertGreater(len(layouts), 1)

    def test_each_employee_gets_the_layout_its_work_needs(self):
        expected = {
            # 宽统计表，用户要一边看结果一边提下一个需求。
            "data_analyst": LAYOUT_ANALYSIS_SPLIT,
            # 产出是一份正在成形的文档。
            "report_writer": LAYOUT_DOCUMENT_SPLIT,
            # 回答基本是成段文字，第二栏会一直空着。
            "career_coach": LAYOUT_CONVERSATION,
        }
        for employee_id, layout in expected.items():
            with self.subTest(employee=employee_id):
                self.assertEqual(get_employee(employee_id).layout, layout)

    def test_employee_ids_are_unique(self):
        ids = [employee.employee_id for employee in list_employees()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_unknown_employee_is_refused(self):
        with self.assertRaises(EmployeeError):
            get_employee("nobody")
        with self.assertRaises(EmployeeError):
            get_employee("")

    def test_unavailable_employee_cannot_be_started(self):
        self.assertEqual(require_available("data_analyst").employee_id, "data_analyst")
        with self.assertRaises(EmployeeError):
            require_available("career_coach")

    def test_collaborators_must_be_employees_that_exist(self):
        for employee in list_employees():
            for target in employee.can_invoke:
                with self.subTest(caller=employee.employee_id, target=target):
                    self.assertTrue(get_employee(target))

    def test_invocation_is_allowed_only_where_configured(self):
        self.assertTrue(can_invoke("data_analyst", "report_writer"))
        self.assertFalse(can_invoke("data_analyst", "career_coach"))
        self.assertFalse(can_invoke("report_writer", "data_analyst"))

    def test_only_available_employees_declare_skills(self):
        """尚未开放的员工不应该声称自己有能力。"""

        for employee in list_employees():
            if not employee.available:
                with self.subTest(employee=employee.employee_id):
                    self.assertEqual(employee.skills, ())


if __name__ == "__main__":
    unittest.main()
