"""End-to-end cover for the Data Analyst closed loop, through a real browser.

The other tests call Flask's test client, which never runs the page's
JavaScript. Anything that lives only in the browser — the token header on every
request, the blob download of the Word report, the confirm button — is
invisible to them. A broken front end would keep the suite green.

This test starts the real server, drives Chromium through the whole loop
(进入工作区 → 上传 → 追问 → 确认执行 → 下载报告) and writes a screenshot at every
step. In CI those screenshots are uploaded as build artifacts, which is how the
project convention「页面已截图确认」gets satisfied automatically instead of by
someone remembering to do it.

It skips itself when Playwright or a browser is unavailable, so a developer
without them still gets a green local run.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = Path(
    os.environ.get("E2E_SCREENSHOT_DIR", REPO_ROOT / "tests" / "_screenshots")
)
SERVER_START_TIMEOUT = 60
RELIABILITY_REQUIREMENT = "做信度分析，题项是 A1、A2、A3、A4"
FAKE_REPLIES_PATH = REPO_ROOT / "tests" / "_fake_replies.json"
DESCRIPTIVE_REQUIREMENT = "对 X、W、M1、M2、Y 做描述统计"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_health(port: int, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + SERVER_START_TIMEOUT
    url = f"http://127.0.0.1:{port}/api/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"服务进程提前退出，退出码 {process.returncode}：\n"
                f"{(process.stdout.read() if process.stdout else '')}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.4)
    raise RuntimeError("服务在超时时间内没有就绪。")


def launch_chromium(playwright):
    """Use whatever browser is installed, with the flags containers need."""

    args = ["--no-sandbox", "--disable-dev-shm-usage"]
    try:
        return playwright.chromium.launch(args=args)
    except Exception:
        candidates = sorted(
            Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")).glob(
                "chromium-*/chrome-linux/chrome"
            )
        )
        if not candidates:
            raise
        return playwright.chromium.launch(
            executable_path=str(candidates[-1]), args=args
        )


class ClosedLoopBrowserTests(unittest.TestCase):
    """One pass through everything a user actually does."""

    @classmethod
    def setUpClass(cls):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:  # pragma: no cover - depends on the environment
            raise unittest.SkipTest("未安装 playwright，跳过浏览器端到端测试。")

        cls.runtime = tempfile.TemporaryDirectory()
        cls.root = Path(cls.runtime.name)

        # The token has to exist before the server starts serving requests.
        sys.path.insert(0, str(REPO_ROOT))
        from modules.users import UserStore
        from modules.workspace import state_database_path

        store = UserStore(state_database_path(cls.root))
        cls.token = store.issue_token(store.create_user("e2e.analyst").user_id)

        cls.port = free_port()
        environment = {
            **os.environ,
            "DATA_ANALYST_RUNTIME_ROOT": str(cls.root),
            "HOST": "127.0.0.1",
            "PORT": str(cls.port),
            "PYTHONUNBUFFERED": "1",
            # 写作流程要调模型。用假客户端：既不花钱不联网，也能精确控制
            # 模型说什么——要验的是界面和校验回路，不是模型写得好不好。
            "LLM_PROVIDER": "fake",
            "E2E_FAKE_REPLIES": str(FAKE_REPLIES_PATH),
        }
        cls.server = subprocess.Popen(
            [sys.executable, "-m", "modules.web.app"],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_health(cls.port, cls.server)
        except Exception:
            cls.server.kill()
            cls.runtime.cleanup()
            raise

        cls._playwright_context = sync_playwright()
        playwright = cls._playwright_context.__enter__()
        try:
            cls.browser = launch_chromium(playwright)
        except Exception as exc:
            cls._playwright_context.__exit__(None, None, None)
            cls.server.kill()
            cls.runtime.cleanup()
            raise unittest.SkipTest(f"无法启动 Chromium，跳过端到端测试：{exc}")

        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "browser", None):
            cls.browser.close()
        if getattr(cls, "_playwright_context", None):
            cls._playwright_context.__exit__(None, None, None)
        if getattr(cls, "server", None):
            cls.server.terminate()
            try:
                cls.server.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                cls.server.kill()
        if getattr(cls, "runtime", None):
            cls.runtime.cleanup()

    def setUp(self):
        self.page = self.browser.new_page(viewport={"width": 1520, "height": 950})
        self.console_errors: list[str] = []
        self.page.on("pageerror", lambda error: self.console_errors.append(str(error)))
        self.step = 0

    def tearDown(self):
        self.page.close()

    def shot(self, name: str) -> None:
        self.step += 1
        self.page.screenshot(path=SCREENSHOT_DIR / f"{self.step:02d}-{name}.png")

    def plan_and_confirm(self, requirement: str) -> None:
        """Send a requirement and press the confirm button it produces.

        Waiting on ``button.confirm-button`` alone is a race: the previous
        run's button is still in the DOM, disabled, so the wait returns at once
        and the click lands on the wrong one. Wait for the count to grow.
        """

        page = self.page
        before = page.locator("button.confirm-button").count()
        page.fill("#prompt", requirement)
        page.click("#sendButton")
        page.wait_for_function(
            "expected => document.querySelectorAll('button.confirm-button').length >= expected",
            arg=before + 1,
            timeout=60000,
        )
        self.shot("参数齐全待确认")
        page.locator("button.confirm-button").last.click()

    def test_a_user_can_go_from_token_to_downloaded_report(self):
        page = self.page
        page.goto(f"http://127.0.0.1:{self.port}/", wait_until="networkidle")
        self.shot("未登录")
        self.assertEqual(page.inner_text("#authState"), "未登录")

        # 侧栏由 /api/employees 渲染，工作区布局取自当前员工。
        page.wait_for_selector("#employeeList [data-employee]", timeout=15000)
        self.assertGreaterEqual(page.locator("#employeeList [data-employee]").count(), 2)
        self.assertEqual(
            page.get_attribute("#workspace", "data-layout"), "analysis_split"
        )
        self.assertEqual(page.inner_text("#convTitle"), "分析对话")

        # 错误令牌必须被挡住，而且不能把页面搞坏。
        page.fill("#tokenInput", "definitely-not-a-valid-token")
        page.click("#connectButton")
        page.wait_for_timeout(800)
        self.shot("错误令牌被拒绝")
        self.assertEqual(page.inner_text("#authState"), "未登录")
        self.assertIn("令牌无效", page.inner_text("#messages"))

        page.fill("#tokenInput", self.token)
        page.click("#connectButton")
        page.wait_for_selector("#authState:has-text('e2e.analyst')", timeout=15000)
        self.shot("进入工作区")

        page.set_input_files("#fileInput", str(_fixture_csv()))
        page.wait_for_selector("#resultStatus:has-text('可分析')", timeout=60000)
        self.shot("上传与数据质量检查")
        self.assertIn("180 行", page.inner_text("#fileDetail"))

        # 参数不全时必须停下来追问，而不是硬跑。
        page.fill("#prompt", "做线性回归")
        page.click("#sendButton")
        page.wait_for_selector("#resultStatus:has-text('等待补充')", timeout=60000)
        self.shot("参数不全时追问")
        self.assertIn("因变量", page.inner_text("#messages"))

        self.plan_and_confirm(RELIABILITY_REQUIREMENT)
        page.wait_for_selector("#resultStatus:has-text('已完成')", timeout=120000)
        self.shot("分析结果")
        self.assertIn("Cronbach", page.inner_text("#resultScroll"))

        # 报告下载走的是带令牌的 fetch，只有真浏览器能验证这条路。
        with page.expect_download(timeout=60000) as download:
            page.click("#downloadLink")
        saved = SCREENSHOT_DIR / "数据分析报告.docx"
        download.value.save_as(saved)
        self.shot("报告已下载")
        self.assertGreater(saved.stat().st_size, 1000)

        # 换一个分析，历史里应该出现两条，而不是把上一条覆盖掉。
        self.plan_and_confirm(DESCRIPTIVE_REQUIREMENT)
        page.wait_for_selector("#resultScroll:has-text('描述统计')", timeout=120000)
        page.wait_for_function(
            "() => document.querySelectorAll('#historyItems [data-job]').length === 2",
            timeout=30000,
        )
        self.shot("会话内分析历史")

        # 点回第一次分析：结果和报告链接都要跟着切回去。
        first_entry = page.locator("#historyItems [data-job]").last
        first_job = first_entry.get_attribute("data-job")
        first_entry.click()
        page.wait_for_selector("#resultScroll:has-text('Cronbach')", timeout=60000)
        self.shot("切回之前的分析")
        self.assertEqual(
            page.locator("#historyItems [data-job].active").get_attribute("data-job"),
            first_job,
        )
        self.assertIn("已切回之前的分析", page.inner_text("#messages"))

        # 完全相同的请求必须命中任务复用，不重新计算也不生成第二份报告。
        self.plan_and_confirm(RELIABILITY_REQUIREMENT)
        page.wait_for_selector("#messages:has-text('没有重新计算')", timeout=120000)
        self.shot("重复请求命中复用")

        # 两次不同的分析各一份报告；重复的那次不应该再生成第三份。
        reports = list((self.root / "sessions").glob("*/reports/*.docx"))
        self.assertEqual(len(reports), 2, "重复的相同分析不应该生成新的报告。")
        self.assertEqual(
            page.locator("#historyItems [data-job]").count(),
            2,
            "重复的相同分析不应该在历史里多出一条。",
        )

        self.assertEqual(self.console_errors, [], "页面在使用过程中抛出了 JS 错误。")


def _fixture_csv() -> Path:
    """Reuse the golden dataset so the browser run analyses known numbers."""

    return REPO_ROOT / "tests" / "golden" / "baseline_survey.csv"


if __name__ == "__main__":
    unittest.main()


class WritingFlowBrowserTests(ClosedLoopBrowserTests):
    """写作类员工：选员工 → 提需求 → 出骨架 → 确认 → 成稿 → 看校验。

    继承同一套服务与浏览器夹具，但走的是另一条路，顺便验证两条路互不干扰。
    """

    def test_a_user_can_go_from_a_brief_to_a_checked_draft(self):
        page = self.page
        page.goto(f"http://127.0.0.1:{self.port}/", wait_until="networkidle")
        page.wait_for_selector("#employeeList [data-employee='prd_writer']", timeout=15000)

        # 选 PRD撰写，工作区应切成文档型布局
        page.click("[data-employee='prd_writer']")
        self.assertEqual(
            page.get_attribute("#workspace", "data-layout"), "document_split"
        )
        self.shot("选中PRD撰写")

        page.fill("#tokenInput", self.token)
        page.click("#connectButton")
        page.wait_for_selector("#authState:has-text('e2e.analyst')", timeout=15000)
        # 文档型员工不需要统计分析的快捷词
        self.assertTrue(page.locator(".chips").is_hidden())
        # 对话栏文案也要跟着员工走，不能写死成数据分析师的
        self.assertEqual(page.inner_text("#convTitle"), "撰写对话")
        self.assertIn("骨架", page.inner_text("#convHint"))
        self.assertIn("PRD撰写", page.inner_text("#welcome"))
        self.assertNotIn("Excel", page.inner_text("#welcome"))

        page.fill("#prompt", "客服每天手工整理订单明细，做个批量导出功能")
        page.click("#sendButton")
        page.wait_for_selector("#docStatus:has-text('骨架待确认')", timeout=60000)
        self.shot("骨架待确认")
        self.assertIn("需求一句话", page.inner_text("#docSkeletonBody"))

        page.click("#confirmSkeleton")
        page.wait_for_selector("#docStatus:has-text('待修改')", timeout=90000)
        self.shot("成稿与校验结果")

        # 这版稿子故意不合格，校验结果必须显示出来
        findings = page.locator("#docFindingsBody .finding")
        self.assertGreater(findings.count(), 0)
        self.assertTrue(page.locator("#docFindingsBody .finding.error").count() > 0)
        self.assertIn("导出", page.inner_text("#docDraftBody"))

        self.assertEqual(self.console_errors, [], "页面在使用过程中抛出了 JS 错误。")
