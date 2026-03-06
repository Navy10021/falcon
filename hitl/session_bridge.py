"""
hitl/session_bridge.py
======================
HITL 웹 인터페이스 ↔ Phase 3 학습 루프 실시간 연동 브리지 (Queue 기반)

아키텍처::

    Phase 3 train loop (메인 스레드)
        ↓ put_options()          ↑ get_selection() (블로킹, timeout=60s)
    ─────────────────────────────────────────────────────
    선택 이벤트 큐 (thread-safe Queue)
    ─────────────────────────────────────────────────────
        ↑ put_selection()        ↓ get_options() (비블로킹)
    web_interface 서버 스레드 (Flask)

사용 예시::

    # train.py Phase 3에서
    bridge = HITLSessionBridge()
    bridge.start_web_server(port=8050)

    # 각 에피소드마다
    bridge.put_options(ep_idx, options)
    selection = bridge.get_selection(timeout=60.0)  # 지휘관 선택 대기
    if selection is None:
        # 타임아웃 → AI 추천 사용
        selection = options[0].option_id

    # web_interface에서 (Flask route 핸들러)
    bridge.put_selection(strategy_id, feedback)

학술 연구용 합성 데이터
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("falcon")


# ──────────────────────────────────────────────
# 이벤트 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class OptionsEvent:
    """Phase 3 → Web: 새 Pareto 후보 전달"""
    episode_idx: int
    options: List[Any]          # ParetoStrategyOption 리스트
    ranked_options: List[Any]   # 선호도 기반 순위 재정렬 후보
    constraints: Any = None


@dataclass
class SelectionEvent:
    """Web → Phase 3: 지휘관 선택 전달"""
    episode_idx: int
    strategy_id: str            # "strategy_0" ~ "strategy_N"
    option_index: int           # ranked_options 내 인덱스
    feedback_rating: int = 5    # 1-5
    feedback_text: str = ""


# ──────────────────────────────────────────────
# 브리지 (싱글톤 패턴 — 한 프로세스 내 단일 인스턴스)
# ──────────────────────────────────────────────

_bridge_instance: Optional["HITLSessionBridge"] = None
_bridge_lock = threading.RLock()  # RLock: __init__ 내부에서 __del__ 재진입 허용


def get_bridge() -> Optional["HITLSessionBridge"]:
    """현재 활성 브리지 인스턴스 반환 (없으면 None)"""
    return _bridge_instance


class HITLSessionBridge:
    """
    Phase 3 학습 루프와 HITL 웹 인터페이스 사이의 실시간 연동 브리지.

    - `options_queue`: Phase 3 → Web 방향 (Pareto 후보 전달)
    - `selection_queue`: Web → Phase 3 방향 (지휘관 선택 전달)
    - 웹 서버는 선택적 백그라운드 스레드로 기동 (Flask 필요)
    """

    def __init__(self, auto_timeout: float = 60.0):
        """
        Parameters
        ----------
        auto_timeout : float
            get_selection() 에서 지휘관 응답 대기 최대 시간 (초).
            타임아웃 시 None 반환 → 학습 루프가 AI 추천을 자동 사용.
        """
        self.auto_timeout = auto_timeout
        self._options_queue: queue.Queue = queue.Queue(maxsize=1)
        self._selection_queue: queue.Queue = queue.Queue(maxsize=1)
        self._server_thread: Optional[threading.Thread] = None
        self._server_running = threading.Event()
        self._current_episode: int = -1

        # 전역 인스턴스 등록
        global _bridge_instance
        with _bridge_lock:
            _bridge_instance = self

        logger.info("[HITLBridge] 초기화 완료 (auto_timeout=%.1fs)", auto_timeout)

    # ──────────────────────────────────────────
    # Phase 3 → Web 방향
    # ──────────────────────────────────────────

    def put_options(
        self,
        episode_idx: int,
        options: List[Any],
        ranked_options: List[Any],
        constraints: Any = None,
    ) -> None:
        """
        Phase 3 루프에서 호출: 현재 에피소드의 Pareto 후보를 웹 UI로 전달.
        이전 미소비 후보가 있으면 버리고 새 것으로 교체.
        """
        event = OptionsEvent(
            episode_idx=episode_idx,
            options=options,
            ranked_options=ranked_options,
            constraints=constraints,
        )
        self._current_episode = episode_idx

        # 이전 미소비 이벤트 제거
        try:
            self._options_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._options_queue.put_nowait(event)
            logger.debug("[HITLBridge] EP %d: %d개 후보 전달", episode_idx, len(options))
        except queue.Full:
            pass  # UI가 느릴 경우 조용히 드롭

    def get_options_nowait(self) -> Optional[OptionsEvent]:
        """Web 서버에서 호출: 가장 최신 Pareto 후보를 즉시 반환 (없으면 None)"""
        try:
            return self._options_queue.get_nowait()
        except queue.Empty:
            return None

    # ──────────────────────────────────────────
    # Web → Phase 3 방향
    # ──────────────────────────────────────────

    def put_selection(
        self,
        episode_idx: int,
        strategy_id: str,
        option_index: int = 0,
        feedback_rating: int = 5,
        feedback_text: str = "",
    ) -> bool:
        """
        Web 서버에서 호출: 지휘관이 전략을 선택했을 때.

        Returns
        -------
        bool
            True이면 큐에 성공적으로 넣음, False이면 Phase 3가 이미 다른 에피소드로 넘어감.
        """
        if episode_idx != self._current_episode:
            logger.warning(
                "[HITLBridge] 에피소드 불일치: 현재=%d, 수신=%d (무시)",
                self._current_episode, episode_idx,
            )
            return False

        event = SelectionEvent(
            episode_idx=episode_idx,
            strategy_id=strategy_id,
            option_index=option_index,
            feedback_rating=feedback_rating,
            feedback_text=feedback_text,
        )

        # 이전 미소비 선택 제거 후 최신 선택 반영
        try:
            self._selection_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._selection_queue.put_nowait(event)
            logger.info(
                "[HITLBridge] EP %d: 지휘관 선택 수신 — %s (rating=%d)",
                episode_idx, strategy_id, feedback_rating,
            )
            return True
        except queue.Full:
            return False

    def get_selection(self, timeout: Optional[float] = None) -> Optional[SelectionEvent]:
        """
        Phase 3 루프에서 호출: 지휘관 선택 대기.

        Parameters
        ----------
        timeout : float, optional
            None이면 self.auto_timeout 사용.

        Returns
        -------
        SelectionEvent or None
            지휘관이 선택했으면 SelectionEvent, 타임아웃이면 None.
        """
        t = timeout if timeout is not None else self.auto_timeout
        try:
            return self._selection_queue.get(timeout=t)
        except queue.Empty:
            logger.debug("[HITLBridge] EP %d: 타임아웃 (%.1fs) → AI 추천 자동 적용",
                         self._current_episode, t)
            return None

    # ──────────────────────────────────────────
    # 상태 조회
    # ──────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """현재 브리지 상태 딕셔너리"""
        return {
            "current_episode": self._current_episode,
            "options_pending": not self._options_queue.empty(),
            "selection_pending": not self._selection_queue.empty(),
            "server_running": self._server_running.is_set(),
            "auto_timeout": self.auto_timeout,
        }

    # ──────────────────────────────────────────
    # 웹 서버 관리
    # ──────────────────────────────────────────

    def start_web_server(self, port: int = 8050, host: str = "127.0.0.1") -> bool:
        """
        Flask HITL 웹 서버를 백그라운드 스레드로 기동.

        Flask 미설치 시 False 반환 (학습 루프는 계속 진행).
        """
        if self._server_thread is not None and self._server_thread.is_alive():
            logger.info("[HITLBridge] 웹 서버가 이미 실행 중")
            return True

        def _run_server():
            try:
                from hitl.web_interface import create_falcon_app
                app = create_falcon_app(bridge=self)
                if app is None:
                    logger.warning("[HITLBridge] Flask 앱 생성 실패 (Flask 미설치?)")
                    return
                self._server_running.set()
                logger.info("[HITLBridge] 웹 서버 기동: http://%s:%d", host, port)
                app.run(host=host, port=port, debug=False, use_reloader=False)
            except Exception as exc:
                logger.warning("[HITLBridge] 웹 서버 기동 실패: %s", exc)
            finally:
                self._server_running.clear()

        self._server_thread = threading.Thread(target=_run_server, daemon=True, name="hitl-web")
        self._server_thread.start()

        # 최대 3초 대기 후 서버 기동 확인
        self._server_running.wait(timeout=3.0)
        return self._server_running.is_set()

    def stop_web_server(self) -> None:
        """웹 서버 정리 (daemon 스레드이므로 프로세스 종료 시 자동 종료됨)"""
        self._server_running.clear()
        logger.info("[HITLBridge] 웹 서버 종료 신호 전송")

    def __del__(self):
        global _bridge_instance
        with _bridge_lock:
            if _bridge_instance is self:
                _bridge_instance = None
