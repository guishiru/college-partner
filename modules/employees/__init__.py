"""Virtual employee boundary: identity, Skills, collaborators and layout."""

from modules.employees.service import (
    LAYOUT_ANALYSIS_SPLIT,
    LAYOUT_CONVERSATION,
    LAYOUT_DOCUMENT_SPLIT,
    LAYOUTS,
    Employee,
    EmployeeError,
    can_invoke,
    get_employee,
    list_employees,
    require_available,
)

__all__ = [
    "LAYOUTS",
    "LAYOUT_ANALYSIS_SPLIT",
    "LAYOUT_CONVERSATION",
    "LAYOUT_DOCUMENT_SPLIT",
    "Employee",
    "EmployeeError",
    "can_invoke",
    "get_employee",
    "list_employees",
    "require_available",
]
