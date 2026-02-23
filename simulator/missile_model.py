"""
missile_model.py
================
미사일 궤적·요격 모델 (Missile Trajectory & Intercept Model)

지원 미사일 유형:
  - 탄도미사일 (Ballistic Missile): 포물선 궤적, MRBM/IRBM 급
  - 순항미사일 (Cruise Missile)  : 저고도 지형 추종, 높은 CEP 정밀도
  - 대함미사일 (Anti-Ship Missile): 시커 유도, 수상함 표적
  - SAM/요격 (Interceptor)       : 히트-투-킬 요격 확률 계산

핵심 모델:
  MissileEngagement  : 단일 발사-요격 교환 계산
  SalvoInterceptModel: 다발 살보 요격 (포화 공격 대응)
  BallisticTrajectory: 탄도 궤적 물리 계산 (단순화)

참고:
  - CEP (Circular Error Probable): 50% 착탄 반경
  - Pk (Probability of Kill)     : 요격 확률
  - Pss (Single Shot Kill Prob)  : 단발 요격 Pk
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ontology.combat_schema import Position, Unit, UnitType


# ──────────────────────────────────────────────────────────────
# 열거형
# ──────────────────────────────────────────────────────────────

class MissileType(Enum):
    """미사일 유형"""
    BALLISTIC_SRBM   = "srbm"   # 단거리 탄도 (<1000 km)
    BALLISTIC_MRBM   = "mrbm"   # 중거리 탄도 (1000~3000 km)
    BALLISTIC_IRBM   = "irbm"   # 중장거리 탄도 (3000~5500 km)
    CRUISE_SUBSONIC  = "cruise_sub"   # 아음속 순항 (0.8M)
    CRUISE_SUPERSONIC = "cruise_sup"  # 초음속 순항 (2.5M)
    ANTI_SHIP        = "anti_ship"    # 대함 미사일 (시커 유도)
    AIR_TO_AIR_SR    = "aam_sr"       # 단거리 공대공 (AIM-9X 계열)
    AIR_TO_AIR_LR    = "aam_lr"       # 장거리 공대공 (AIM-120 계열)
    SAM_SHORT        = "sam_short"    # 단거리 SAM (SHORAD)
    SAM_MEDIUM       = "sam_medium"   # 중거리 SAM (천궁·PAC-2)
    SAM_LONG         = "sam_long"     # 장거리 SAM (PAC-3·SM-3)


class InterceptResult(Enum):
    """요격 결과"""
    KILL        = "kill"        # 요격 성공 (파괴)
    DAMAGED     = "damaged"     # 부분 파괴 (기능 저하)
    MISS        = "miss"        # 요격 실패
    NO_INTERCEPT = "no_intercept"  # 요격 불가 (사거리·속도 초과)


# ──────────────────────────────────────────────────────────────
# 미사일 성능 테이블 (학술 합성 데이터)
# ──────────────────────────────────────────────────────────────

@dataclass
class MissileSpec:
    """미사일 제원"""
    missile_type: MissileType
    range_km: float             # 최대 사거리 (km)
    speed_mach: float           # 속도 (마하)
    cep_m: float                # CEP (원형공산오차, m)
    warhead_kg: float           # 탄두 중량 (kg)
    maneuverability: float      # 기동성 [0, 1] (요격 어려움)
    launch_ready_sec: float     # 발사 준비 시간 (초)


MISSILE_SPECS: Dict[MissileType, MissileSpec] = {
    MissileType.BALLISTIC_SRBM: MissileSpec(
        MissileType.BALLISTIC_SRBM, 800, 5.0, 300.0, 500.0, 0.1, 900),
    MissileType.BALLISTIC_MRBM: MissileSpec(
        MissileType.BALLISTIC_MRBM, 2000, 7.0, 250.0, 1000.0, 0.1, 1800),
    MissileType.BALLISTIC_IRBM: MissileSpec(
        MissileType.BALLISTIC_IRBM, 4500, 10.0, 200.0, 2000.0, 0.1, 3600),
    MissileType.CRUISE_SUBSONIC: MissileSpec(
        MissileType.CRUISE_SUBSONIC, 1500, 0.8, 10.0, 500.0, 0.7, 300),
    MissileType.CRUISE_SUPERSONIC: MissileSpec(
        MissileType.CRUISE_SUPERSONIC, 500, 2.5, 5.0, 300.0, 0.8, 180),
    MissileType.ANTI_SHIP: MissileSpec(
        MissileType.ANTI_SHIP, 250, 0.9, 8.0, 500.0, 0.6, 120),
    MissileType.AIR_TO_AIR_SR: MissileSpec(
        MissileType.AIR_TO_AIR_SR, 35, 3.0, 9.0, 11.0, 0.9, 5),
    MissileType.AIR_TO_AIR_LR: MissileSpec(
        MissileType.AIR_TO_AIR_LR, 180, 4.0, 5.0, 23.0, 0.6, 10),
    MissileType.SAM_SHORT: MissileSpec(
        MissileType.SAM_SHORT, 20, 3.0, 5.0, 20.0, 0.85, 30),
    MissileType.SAM_MEDIUM: MissileSpec(
        MissileType.SAM_MEDIUM, 100, 4.5, 3.0, 80.0, 0.70, 60),
    MissileType.SAM_LONG: MissileSpec(
        MissileType.SAM_LONG, 300, 6.0, 1.0, 100.0, 0.50, 90),
}

# SAM 요격 대상 미사일 유형별 단발 요격 확률 (Pss)
SAM_PSS: Dict[Tuple[MissileType, MissileType], float] = {
    # (요격체, 목표) : Pss
    (MissileType.SAM_SHORT,  MissileType.CRUISE_SUBSONIC): 0.70,
    (MissileType.SAM_SHORT,  MissileType.CRUISE_SUPERSONIC): 0.55,
    (MissileType.SAM_SHORT,  MissileType.ANTI_SHIP): 0.65,
    (MissileType.SAM_SHORT,  MissileType.AIR_TO_AIR_SR): 0.50,
    (MissileType.SAM_MEDIUM, MissileType.BALLISTIC_SRBM): 0.60,
    (MissileType.SAM_MEDIUM, MissileType.CRUISE_SUBSONIC): 0.80,
    (MissileType.SAM_MEDIUM, MissileType.CRUISE_SUPERSONIC): 0.65,
    (MissileType.SAM_MEDIUM, MissileType.ANTI_SHIP): 0.75,
    (MissileType.SAM_LONG,   MissileType.BALLISTIC_SRBM): 0.80,
    (MissileType.SAM_LONG,   MissileType.BALLISTIC_MRBM): 0.65,
    (MissileType.SAM_LONG,   MissileType.BALLISTIC_IRBM): 0.45,
    (MissileType.SAM_LONG,   MissileType.CRUISE_SUBSONIC): 0.85,
    (MissileType.SAM_LONG,   MissileType.CRUISE_SUPERSONIC): 0.72,
    (MissileType.AIR_TO_AIR_SR, MissileType.CRUISE_SUBSONIC): 0.60,
    (MissileType.AIR_TO_AIR_LR, MissileType.BALLISTIC_SRBM): 0.40,
    (MissileType.AIR_TO_AIR_LR, MissileType.CRUISE_SUBSONIC): 0.75,
}
DEFAULT_PSS = 0.20  # 미등재 조합 기본값


# ──────────────────────────────────────────────────────────────
# 탄도 궤적 계산 (단순화)
# ──────────────────────────────────────────────────────────────

@dataclass
class BallisticPoint:
    """탄도 경로점"""
    t_sec: float      # 경과 시간 (초)
    x_km: float       # 수평 거리 (km)
    alt_km: float     # 고도 (km)
    velocity_mps: float  # 속도 (m/s)


class BallisticTrajectory:
    """
    단순화 탄도 궤적 계산 (진공 포물선 근사 + 대기 항력 보정).
    실전 수준 정확도가 아닌 학술 연구용 근사 모델.
    """

    G     = 9.81        # 중력 가속도 (m/s²)
    R_KM  = 6371.0      # 지구 반경 (km)

    def compute(
        self,
        spec: MissileSpec,
        range_km: float,
        dt_sec: float = 10.0,
    ) -> List[BallisticPoint]:
        """
        탄도 궤적 계산 → 경로점 목록 반환.
        단순 포물선 모델: 최대 고도 ≈ range/4 (MRBM 경험식).
        """
        if range_km > spec.range_km:
            range_km = spec.range_km

        # 발사각 최적화 (45도 근사, 장거리는 더 높음)
        v0_mps = spec.speed_mach * 340.0  # 마하 → m/s
        range_m = range_km * 1000.0

        # 비행 시간 T (초)
        t_flight = range_m / (v0_mps * 0.85)  # 0.85: 수평 속도 근사 계수

        # 최대 고도 (경험식: SRBM ~ 100km, MRBM ~ 500km, IRBM ~ 1200km)
        apex_km = range_km * {
            MissileType.BALLISTIC_SRBM: 0.12,
            MissileType.BALLISTIC_MRBM: 0.25,
            MissileType.BALLISTIC_IRBM: 0.27,
        }.get(spec.missile_type, 0.05)

        trajectory: List[BallisticPoint] = []
        t = 0.0
        while t <= t_flight:
            frac = t / t_flight
            x_km = range_km * frac
            # 포물선 고도: h = 4 × apex × frac × (1-frac)
            alt_km = float(4.0 * apex_km * frac * (1.0 - frac))
            vel_mps = v0_mps * (1.0 - 0.3 * frac)  # 대기 마찰 단순화
            trajectory.append(BallisticPoint(t, x_km, alt_km, vel_mps))
            t += dt_sec

        return trajectory

    def time_of_flight_sec(self, spec: MissileSpec, range_km: float) -> float:
        """비행 시간 추정 (초)"""
        v0 = spec.speed_mach * 340.0
        return (min(range_km, spec.range_km) * 1000.0) / (v0 * 0.85)


# ──────────────────────────────────────────────────────────────
# 단발 교전 모델
# ──────────────────────────────────────────────────────────────

@dataclass
class MissileEngagement:
    """단일 미사일 발사-요격 교전 결과"""
    missile_type: MissileType
    interceptor_type: Optional[MissileType]
    launch_position: Position
    target_position: Position
    range_km: float
    time_of_flight_sec: float
    intercept_result: InterceptResult
    pk: float                           # 요격 성공 확률 (사후)
    impact_cep_m: float                 # 착탄 CEP (요격 실패 시)
    damage_radius_m: float              # 피해 반경 (탄두 중량 기반)


class MissileModel:
    """
    미사일 발사-요격 시뮬레이션.
    발사 계산, 요격 확률, 착탄 효과를 통합 관리.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self._traj = BallisticTrajectory()

    def _damage_radius(self, warhead_kg: float) -> float:
        """탄두 중량 → 피해 반경 (m), Hopkinson-Cranz 스케일링 단순화"""
        return 50.0 * (warhead_kg ** (1.0 / 3.0))

    def _intercept_pk(
        self,
        missile_type: MissileType,
        interceptor_type: Optional[MissileType],
        n_interceptors: int = 1,
        ew_support: float = 0.0,   # 전자전 지원 [0, 1]
    ) -> float:
        """
        요격 확률 계산.
        여러 요격체: Pk_total = 1 - (1-Pss)^n
        EW 지원: Pss 보정 ×(1 + ew_support × 0.2)
        """
        if interceptor_type is None:
            return 0.0
        pss = SAM_PSS.get((interceptor_type, missile_type), DEFAULT_PSS)
        # 미사일 기동성에 따른 감쇠
        spec = MISSILE_SPECS[missile_type]
        pss *= (1.0 - spec.maneuverability * 0.3)
        # EW 지원 보정
        pss = min(0.97, pss * (1.0 + ew_support * 0.2))
        # 다발 요격 (n개 동시 발사)
        pk = 1.0 - (1.0 - pss) ** n_interceptors
        return float(np.clip(pk, 0.0, 0.97))

    def simulate_launch(
        self,
        missile_type: MissileType,
        launch_pos: Position,
        target_pos: Position,
        interceptor_type: Optional[MissileType] = None,
        n_interceptors: int = 1,
        ew_support: float = 0.0,
    ) -> MissileEngagement:
        """
        미사일 발사 → 요격 시도 → 결과 반환.
        """
        spec = MISSILE_SPECS[missile_type]
        range_km = launch_pos.distance_to(target_pos)

        # 사거리 초과 여부
        if range_km > spec.range_km:
            return MissileEngagement(
                missile_type     = missile_type,
                interceptor_type = interceptor_type,
                launch_position  = launch_pos,
                target_position  = target_pos,
                range_km         = range_km,
                time_of_flight_sec = 0.0,
                intercept_result = InterceptResult.NO_INTERCEPT,
                pk               = 0.0,
                impact_cep_m     = spec.cep_m,
                damage_radius_m  = self._damage_radius(spec.warhead_kg),
            )

        tof = self._traj.time_of_flight_sec(spec, range_km)
        pk  = self._intercept_pk(
            missile_type, interceptor_type, n_interceptors, ew_support
        )

        roll = self.rng.random()
        if roll < pk:
            result = InterceptResult.KILL
        elif roll < pk + 0.05:        # 5% 부분 파괴 마진
            result = InterceptResult.DAMAGED
        else:
            result = InterceptResult.MISS

        # 착탄 CEP (요격 실패 시)
        cep = spec.cep_m * self.rng.uniform(0.8, 1.2) if result == InterceptResult.MISS else 0.0

        return MissileEngagement(
            missile_type       = missile_type,
            interceptor_type   = interceptor_type,
            launch_position    = launch_pos,
            target_position    = target_pos,
            range_km           = range_km,
            time_of_flight_sec = tof,
            intercept_result   = result,
            pk                 = pk,
            impact_cep_m       = float(cep),
            damage_radius_m    = self._damage_radius(spec.warhead_kg),
        )


# ──────────────────────────────────────────────────────────────
# 포화 공격 대응 모델 (Salvo Intercept)
# ──────────────────────────────────────────────────────────────

@dataclass
class SalvoInterceptReport:
    """다발 살보 요격 보고서"""
    total_missiles_fired: int
    intercepted_count: int
    leaked_count: int           # 방어 돌파 미사일 수
    intercept_rate: float       # 요격율 [0, 1]
    saturation_threshold: int   # 포화 공격 임계 (이 이상이면 방어 붕괴)
    is_saturated: bool          # 포화 공격 여부
    engagements: List[MissileEngagement] = field(default_factory=list)


class SalvoInterceptModel:
    """
    포화 공격 대응 모델.
    방어 포대 수 × 발사 능력이 한계를 넘으면 방어 붕괴.
    """

    def __init__(self, seed: Optional[int] = None):
        self.model = MissileModel(seed=seed)

    def simulate_salvo(
        self,
        missile_type: MissileType,
        n_missiles: int,
        interceptor_type: MissileType,
        n_interceptor_batteries: int,
        shots_per_battery: int = 2,    # 포대당 동시 발사 수
        launch_pos: Optional[Position] = None,
        target_pos: Optional[Position] = None,
        ew_support: float = 0.0,
    ) -> SalvoInterceptReport:
        """
        n_missiles 살보 발사 → 방어 포대 요격 시뮬레이션.
        포대가 감당할 수 있는 미사일 = n_batteries × shots_per_battery.
        초과 미사일 → 방어 불가 (돌파).
        """
        if launch_pos is None:
            launch_pos = Position(0.0, 0.0)
        if target_pos is None:
            target_pos = Position(100.0, 0.0)

        max_intercepts = n_interceptor_batteries * shots_per_battery
        saturation_threshold = max_intercepts

        engagements: List[MissileEngagement] = []
        intercepted = 0

        for i in range(n_missiles):
            if i < max_intercepts:
                n_int = 1
            else:
                n_int = 0  # 포화 → 요격 불가

            eng = self.model.simulate_launch(
                missile_type     = missile_type,
                launch_pos       = launch_pos,
                target_pos       = target_pos,
                interceptor_type = interceptor_type if n_int > 0 else None,
                n_interceptors   = n_int,
                ew_support       = ew_support,
            )
            engagements.append(eng)
            if eng.intercept_result == InterceptResult.KILL:
                intercepted += 1

        leaked = n_missiles - intercepted

        return SalvoInterceptReport(
            total_missiles_fired  = n_missiles,
            intercepted_count     = intercepted,
            leaked_count          = leaked,
            intercept_rate        = float(intercepted) / max(n_missiles, 1),
            saturation_threshold  = saturation_threshold,
            is_saturated          = n_missiles > saturation_threshold,
            engagements           = engagements,
        )


# ──────────────────────────────────────────────────────────────
# 자가 테스트
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ontology.combat_schema import Position

    print("=== Missile Model Test ===")
    model = MissileModel(seed=42)
    launch = Position(0.0, 0.0)
    target = Position(300.0, 0.0)   # 300km 거리

    for m_type, int_type in [
        (MissileType.CRUISE_SUBSONIC, MissileType.SAM_MEDIUM),
        (MissileType.BALLISTIC_SRBM,  MissileType.SAM_LONG),
        (MissileType.ANTI_SHIP,       MissileType.SAM_SHORT),
    ]:
        eng = model.simulate_launch(m_type, launch, target, int_type, n_interceptors=2)
        print(f"  {m_type.value:20s} vs {int_type.value:12s} "
              f"→ {eng.intercept_result.value:12s}  Pk={eng.pk:.2f}  "
              f"ToF={eng.time_of_flight_sec:.0f}s")

    print("\n=== Saturation Attack Test ===")
    salvo_model = SalvoInterceptModel(seed=42)
    for n_m in [4, 8, 16]:
        report = salvo_model.simulate_salvo(
            MissileType.CRUISE_SUBSONIC, n_m,
            MissileType.SAM_MEDIUM, n_interceptor_batteries=2
        )
        print(f"  {n_m:2d}발 발사 → 요격{report.intercepted_count}발 / "
              f"돌파{report.leaked_count}발  포화={'O' if report.is_saturated else 'X'}")

    print("\n=== Ballistic Trajectory Test ===")
    traj = BallisticTrajectory()
    spec = MISSILE_SPECS[MissileType.BALLISTIC_MRBM]
    pts  = traj.compute(spec, 1500.0, dt_sec=60.0)
    print(f"  MRBM 1500km: {len(pts)}점  최대고도 {max(p.alt_km for p in pts):.0f}km  "
          f"ToF {traj.time_of_flight_sec(spec, 1500.0)/60:.1f}분")
