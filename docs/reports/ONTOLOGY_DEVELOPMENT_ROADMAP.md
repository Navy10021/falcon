# FALCON 온톨로지 발전 로드맵

**작성일**: 2026-02-22
**대상**: 현재 `ontology/` 모듈 전반 + 연동 시뮬레이터/RL 모듈
**목표**: 육·해·공군·해병대 현실 전력을 반영하고, 다차원 전장 개념(사이버·우주·드론·미사일)을 온톨로지에 통합

---

## 1. 현황 분석 (As-Is)

### 1-1. 현재 UnitType (8종)

| 유형 | 설명 |
|------|------|
| INFANTRY | 보병 |
| ARMOR | 장갑/기갑 |
| ARTILLERY | 포병 |
| AVIATION | 항공 (지상군 항공) |
| ENGINEER | 공병 |
| SIGNAL | 신호/통신 |
| LOGISTICS | 군수/보급 |
| ELECTRONIC_WARFARE | 전자전 |

### 1-2. 현재 한계

| 한계 영역 | 구체적 문제 |
|-----------|-------------|
| **군종 구분 없음** | 육군·해군·공군·해병대 개념 미분리 — ForceAlignment(BLUE/RED)만 존재 |
| **해군 전력 부재** | 수상함·잠수함·상륙함 개념 없음 |
| **공군 전력 미세화 부족** | AVIATION이 전술기·전략기·무인기를 모두 포괄, 구분 불가 |
| **해병대 부재** | 상륙작전 특수 유닛 없음 |
| **미사일 개념 없음** | 탄도·순항·대함·방공 미사일 미존재 |
| **드론 미분리** | 무인기가 AVIATION에 혼재 |
| **사이버 전력 없음** | 사이버전 유닛·효과 미존재 |
| **우주 전력 없음** | 위성·우주 자산 미존재 |
| **지휘 계층 없음** | 사단-연대-대대 계층 구조 미반영 |
| **병력 수 비현실적** | headcount ∈ [60, 200] — 실제 대대(700~900명), 사단(1.5만명) 규모 미반영 |
| **지형 추상화** | 6종 지형이 해상·우주·사이버 공간 미포함 |

---

## 2. 발전 방향 및 단계별 로드맵

---

### STEP 1 — 군종(Branch) 분리 + 지휘 계층(Command Hierarchy) 도입

**우선순위**: 즉시 (가장 큰 구조적 개선)

#### 2-1. BranchType Enum 신설

```python
class BranchType(Enum):
    """군종(군/병과)"""
    ARMY          = "army"           # 육군
    NAVY          = "navy"           # 해군
    AIR_FORCE     = "air_force"      # 공군
    MARINE_CORPS  = "marine_corps"   # 해병대
    STRATEGIC     = "strategic"      # 전략사령부 (합참 직속)
    SPACE         = "space"          # 우주군 (신설)
    CYBER         = "cyber"          # 사이버사령부
```

#### 2-2. EchelonType Enum 신설 (지휘 계층)

```python
class EchelonType(Enum):
    """부대 계층 (NATO 표준)"""
    # 육군
    SQUAD         = "squad"          # 분대 (8~13명)
    PLATOON       = "platoon"        # 소대 (30~50명)
    COMPANY       = "company"        # 중대 (80~200명)
    BATTALION     = "battalion"      # 대대 (500~1,000명)
    REGIMENT      = "regiment"       # 연대 (2,000~3,000명)
    BRIGADE       = "brigade"        # 여단 (3,000~5,000명)
    DIVISION      = "division"       # 사단 (10,000~20,000명)
    CORPS         = "corps"          # 군단 (40,000~80,000명)
    ARMY_GROUP    = "army_group"     # 야전군 (100,000+명)

    # 해군
    SHIP          = "ship"           # 개별 함정
    SQUADRON      = "squadron"       # 전대 (함정 3~6척)
    FLEET         = "fleet"          # 함대

    # 공군
    FLIGHT        = "flight"         # 비행편대 (2~4기)
    ELEMENT       = "element"        # 편대 (4~8기)
    WING          = "wing"           # 비행단 (72~100기)
```

#### 2-3. Unit 데이터클래스 확장

```python
@dataclass
class Unit:
    # 기존 필드 유지
    unit_id: str
    unit_type: UnitType
    alignment: ForceAlignment
    position: Position
    capability: Capability
    headcount: int
    status: UnitStatus
    morale: float
    experience: float

    # 신규 필드
    branch: BranchType = BranchType.ARMY    # 군종
    echelon: EchelonType = EchelonType.BATTALION  # 계층
    parent_unit_id: Optional[str] = None    # 상위 부대 ID
    child_unit_ids: List[str] = field(default_factory=list)  # 하위 부대 IDs
    designation: str = ""                   # 부대 명칭 (예: "1대대")
    doctrine_label: str = ""               # 교리 레이블 (예: "기계화보병")
```

---

### STEP 2 — 확장 UnitType (현실 전력 반영)

**우선순위**: 단기 (핵심 기능 완성도)

#### 2-1. 지상군 (육군/해병대) UnitType 확장

```python
# 현행 8종 → 아래로 확장

# 육군 기갑/보병
MECHANIZED_INFANTRY  = "mech_infantry"      # 기계화보병 (IFV 탑승)
AIRBORNE             = "airborne"           # 공수부대
SPECIAL_FORCES       = "special_forces"     # 특수전부대 (707·UDT 계열)
RECON                = "recon"              # 수색정찰 (LRRS 등)
AIR_DEFENSE          = "air_defense"        # 방공포병 (천마·패트리어트)
ROCKET_ARTILLERY     = "rocket_artillery"   # 다연장로켓 (천무·MLRS)
CBRN                 = "cbrn"               # 화생방 부대

# 해병대
MARINE_INFANTRY      = "marine_infantry"    # 해병 보병
AMPHIBIOUS_ASSAULT   = "amphibious_assault" # 상륙돌격 (KAAV 등)
```

#### 2-2. 해군 UnitType 신설

```python
SURFACE_COMBATANT    = "surface_combatant"  # 수상전투함 (구축함·호위함)
SUBMARINE            = "submarine"           # 잠수함 (장보고·손원일급)
AMPHIBIOUS_SHIP      = "amphibious_ship"    # 상륙함 (독도함 계열)
MINE_WARFARE         = "mine_warfare"        # 기뢰전 함정
PATROL_VESSEL        = "patrol_vessel"       # 고속정·초계함
NAVAL_SUPPORT        = "naval_support"       # 군수지원함
COASTAL_DEFENSE      = "coastal_defense"     # 해안방어 (해안포·SSM)
```

#### 2-3. 공군 UnitType 세분화

```python
FIGHTER              = "fighter"             # 전투기 (F-35·F-15K·KF-21)
BOMBER               = "bomber"              # 폭격기 (F-15K Strike 역할)
ISR_AIRCRAFT         = "isr_aircraft"        # 정보·감시·정찰기 (백두·금강)
TANKER               = "tanker"              # 공중급유기 (KC-330)
TRANSPORT_AIRCRAFT   = "transport_aircraft"  # 수송기 (CN-235·C-130)
AEW                  = "aew"                 # 공중조기경보기
```

#### 2-4. 미사일/무인기/전략 UnitType

```python
# 미사일 전력
BALLISTIC_MISSILE    = "ballistic_missile"   # 탄도미사일 (현무-2·3·4)
CRUISE_MISSILE       = "cruise_missile"      # 순항미사일 (현무-3·해성)
ANTI_SHIP_MISSILE    = "anti_ship_missile"   # 대함미사일 (해성·하푼)
SAM_BATTERY          = "sam_battery"         # 지대공 미사일 (천궁·패트리어트)

# 드론/무인기
UAV_RECON            = "uav_recon"           # 정찰 UAV (송골매·RQ-4 계열)
UAV_STRIKE           = "uav_strike"          # 공격 UAV (MQ-9 계열)
LOITERING_MUNITION   = "loitering_munition"  # 배회형 탄약 (자폭드론)
UGV                  = "ugv"                 # 무인지상차량
USV                  = "usv"                 # 무인수상함

# 전략/사이버/우주
CYBER_UNIT           = "cyber_unit"          # 사이버전 부대
SPACE_ASSET          = "space_asset"         # 위성/우주 자산
PSYOPS               = "psyops"              # 심리전 부대
STRATEGIC_MISSILE    = "strategic_missile"   # 전략미사일 (ICBM급 억지)
```

---

### STEP 3 — Capability 확장 (현실 전투 지수 반영)

**우선순위**: 단기

#### 현행 Capability (8개 필드) → 신규 확장 (20+ 필드)

```python
@dataclass
class Capability:
    # 기존 8개 유지 (하위 호환)
    firepower: float           # 화력 지수
    mobility: float            # 기동력
    protection: float          # 방호력
    range_km: float            # 유효 사거리 (km)
    ammo_level: float          # 탄약 수준
    fuel_level: float          # 연료 수준
    comms_quality: float       # 통신 품질
    ew_resistance: float       # 전자전 저항

    # 신규 추가 — 화력 세분화
    anti_armor: float = 0.0    # 대장갑 능력 (0~1)
    anti_air: float = 0.0      # 대공 능력 (0~1)
    indirect_fire: float = 0.0 # 간접화력 능력 (0~1)
    precision_strike: float = 0.0  # 정밀타격 능력 (0~1)

    # 신규 — 해군 전투
    anti_submarine: float = 0.0    # 대잠전 능력
    anti_ship: float = 0.0         # 대함 능력
    mine_laying: float = 0.0       # 기뢰 부설 능력
    amphibious_lift: int = 0       # 상륙 가능 병력 수 (상륙함 전용)

    # 신규 — 공군/항공
    air_to_air: float = 0.0        # 공대공 전투력
    air_to_ground: float = 0.0     # 공대지 공격력
    stealth_factor: float = 0.0    # 스텔스 계수 (0~1)
    combat_radius_km: float = 0.0  # 전투 반경 (km)
    sortie_rate: float = 0.0       # 출격률 (sorties/day)

    # 신규 — 사이버/전자전
    cyber_offense: float = 0.0     # 사이버 공격력
    cyber_defense: float = 0.0     # 사이버 방어력
    ew_offense: float = 0.0        # 전자전 공격 (재밍)

    # 신규 — 지속성
    max_endurance_days: float = 5.0  # 독립 작전 가능 일수
    maintenance_rate: float = 0.85   # 가동률
```

---

### STEP 4 — 현실 병력 규모 테이블 수립

**우선순위**: 단기

#### 4-1. 육군 전형 편제 병력 (대한민국 기준)

| 계층 | 명칭 (예시) | 병력 | UnitType |
|------|------------|------|----------|
| 사단 | 제1보병사단 | 13,000~16,000 | INFANTRY |
| 여단 | 제1전차여단 | 3,500~5,000 | ARMOR |
| 연대 | 제1보병연대 | 2,000~3,000 | INFANTRY |
| 대대 | 제1전차대대 | 700~900 | ARMOR |
| 중대 | 제1보병중대 | 80~150 | INFANTRY |
| 소대 | 제1보병소대 | 30~45 | INFANTRY |
| 포병대대 | 155mm 자주포대대 | 400~600 | ARTILLERY |
| 공병대대 | 제1공병대대 | 500~700 | ENGINEER |

#### 4-2. 해군 전형 편제 (대한민국 기준)

| 함종 | 명칭 (예시) | 승조원 | 배수량 | UnitType |
|------|------------|--------|--------|----------|
| 구축함 DDG | KDX-III 이지스 | 300~400 | 11,000t | SURFACE_COMBATANT |
| 구축함 DD | KDX-II | 220 | 4,500t | SURFACE_COMBATANT |
| 호위함 FF | 인천급 | 140 | 2,300t | SURFACE_COMBATANT |
| 잠수함 SSK | 손원일급(214형) | 27 | 1,700t | SUBMARINE |
| 잠수함 SSK | 장보고급(209형) | 33 | 1,200t | SUBMARINE |
| 상륙함 LPH | 독도함 | 330 | 18,000t | AMPHIBIOUS_SHIP |
| 초계함 PCC | 포항급 | 95 | 1,200t | PATROL_VESSEL |
| 고속정 PKG | 윤영하급 | 40 | 450t | PATROL_VESSEL |

#### 4-3. 공군 전형 편제 (대한민국 기준)

| 부대 | 명칭 (예시) | 항공기 수 | UnitType |
|------|------------|-----------|----------|
| 비행단 | 제11전투비행단 | 72~100기 | FIGHTER |
| 비행전대 | KF-21 전대 | 24기 | FIGHTER |
| 편대 | F-35A 편대 | 8기 | FIGHTER |
| ISR전대 | 백두·금강 | 4~8기 | ISR_AIRCRAFT |
| 급유편대 | KC-330 | 4기 | TANKER |
| 수송편대 | CN-235 | 8~12기 | TRANSPORT_AIRCRAFT |

#### 4-4. 해병대 전형 편제

| 부대 | 병력 | UnitType |
|------|------|----------|
| 해병 사단 | 10,000~12,000 | MARINE_INFANTRY |
| 상륙연대 | 3,000 | MARINE_INFANTRY |
| 상륙대대 | 700~900 | MARINE_INFANTRY |
| 상륙돌격장갑대대 | 350 | AMPHIBIOUS_ASSAULT |

#### 4-5. 미사일 전력 (대한민국 기준)

| 체계 | 사거리 | 탄두 | UnitType |
|------|--------|------|----------|
| 현무-4 탄도미사일 | 800km | 2t | BALLISTIC_MISSILE |
| 현무-3C 순항미사일 | 1,500km | 500kg | CRUISE_MISSILE |
| 해성-II 대함미사일 | 250km | 재래식 | ANTI_SHIP_MISSILE |
| 천궁-II SAM | 40km | 재래식 | SAM_BATTERY |
| 패트리어트 PAC-3 | 20km | 재래식 | SAM_BATTERY |
| 천무 MRLS | 80~160km | 집속/단일탄 | ROCKET_ARTILLERY |

---

### STEP 5 — 공간(Domain) 확장: 사이버·우주·수중

**우선순위**: 중기

#### 5-1. DomainType 확장

```python
class DomainType(Enum):
    # 기존
    GROUND    = "ground"
    AIR       = "air"
    EW        = "electronic_warfare"
    MARITIME  = "maritime"

    # 신규
    SUBSURFACE = "subsurface"      # 수중 (잠수함 작전)
    SPACE      = "space"           # 우주 (위성·우주작전)
    CYBER      = "cyber"           # 사이버 공간
    STRATEGIC  = "strategic"       # 전략 억지 도메인
```

#### 5-2. 사이버 도메인 효과 (CyberEffect)

```python
@dataclass
class CyberEffect:
    """사이버 작전 효과"""
    target_system: str          # 공격 대상 ("C2", "logistics", "sensor", "grid")
    effect_type: str            # "disruption" | "denial" | "deception" | "destruction"
    severity: float             # 효과 크기 [0, 1]
    duration_steps: int         # 지속 스텝 수

    # 교전규칙 준수
    roe_compliant: bool = True  # 사이버 교전규칙 준수 여부
```

#### 5-3. 우주 자산 효과 (SpaceEffect)

```python
@dataclass
class SpaceAssetEffect:
    """우주 자산 제공 효과"""
    asset_type: str         # "imagery_satellite" | "comms_satellite" | "gps" | "early_warning"
    coverage_radius_km: float   # 지원 반경
    revisit_rate_min: float     # 재방문 주기 (분)

    # 효과: 아군 ISR 능력, 통신 품질, 위치 정확도 향상
    isr_bonus: float = 0.0
    comms_bonus: float = 0.0
    precision_bonus: float = 0.0
```

#### 5-4. TerrainType 확장

```python
class TerrainType(Enum):
    # 기존 6종 유지
    OPEN      = "open"
    URBAN     = "urban"
    FOREST    = "forest"
    MOUNTAIN  = "mountain"
    RIVER     = "river"
    COASTAL   = "coastal"

    # 신규
    OCEAN       = "ocean"         # 외해 (해전 구역)
    SHALLOW_SEA = "shallow_sea"   # 내해·연안 (기뢰전·상륙)
    UNDERWATER  = "underwater"    # 수중 (잠수함 작전)
    ARCTIC      = "arctic"        # 극지/동토
    DESERT      = "desert"        # 사막
    JUNGLE      = "jungle"        # 정글
    URBAN_RUBBLE = "urban_rubble" # 폐허화된 시가지 (고강도 전투 후)
    AIRSPACE    = "airspace"      # 공역 구역
    CYBERSPACE  = "cyberspace"    # 사이버 공간 (추상)
```

---

### STEP 6 — 지휘통제(C2) 및 합동작전 온톨로지

**우선순위**: 중기

#### 6-1. CommandStructure 클래스

```python
@dataclass
class CommandStructure:
    """지휘 계층 구조"""
    headquarters_id: str              # 사령부 유닛 ID
    subordinate_ids: List[str]        # 예하 부대 ID 목록
    supported_ids: List[str]          # 지원 관계 부대 ID 목록
    communication_quality: float      # 통신 품질 [0, 1]
    c2_degradation: float = 0.0       # C2 약화 수준 (사이버/EW 공격 시 증가)

    def effective_command_quality(self) -> float:
        """통신 품질 × (1 - c2_degradation)"""
```

#### 6-2. 합동작전 시너지 (JointFiresSupport)

```python
@dataclass
class JointFiresSupport:
    """합동 화력 지원 요청·결과"""
    requester_id: str           # 요청 부대
    supporter_id: str           # 지원 부대
    support_type: str           # "cas" | "naval_gunfire" | "rocket" | "cyber_strike"
    target_location: Position
    effectiveness: float        # 효과 [0, 1]
    response_delay_steps: int   # 응답 지연 (스텝)
    roe_compliant: bool = True
```

---

### STEP 7 — ScenarioFactory 확장: 현실 시나리오 프리셋

**우선순위**: 중기

#### 7-1. 시나리오 프리셋 목록

```python
class ScenarioFactory:

    @staticmethod
    def create_korea_defense_scenario(seed=None) -> CombatKnowledgeGraph:
        """
        한반도 방어작전 시나리오
        - Blue: 육군 사단 + 해군 전단 + 공군 비행단 + 해병 여단
        - Red: 북한 전방군단 (기계화·보병)
        - 규모: 사단급 (1.5만 vs 3만)
        - 지형: 개활지·산악·도시 혼재
        - 특수: 북한 장사정포, 특수전 침투, EMP 위협
        """

    @staticmethod
    def create_amphibious_assault_scenario(seed=None) -> CombatKnowledgeGraph:
        """
        도서/해안 상륙작전 시나리오
        - Blue: 해병 상륙연대 + 해군 함대 + 공군 CAS
        - Red: 도서 방어 보병여단 + 해안방어포
        - 특수: 해상·수중·지상·공중 4차원 연계 작전
        """

    @staticmethod
    def create_air_battle_scenario(seed=None) -> CombatKnowledgeGraph:
        """
        공중전/항공차단 시나리오
        - Blue: F-35·F-15K 비행단 + 방공망
        - Red: 적 전투기 + 탄도미사일 + 방공체계
        - 특수: 공중우세, 전략표적 타격
        """

    @staticmethod
    def create_multidomain_contest_scenario(seed=None) -> CombatKnowledgeGraph:
        """
        다영역 경쟁 시나리오 (지상+해상+공중+사이버+우주)
        - 5개 도메인 동시 운용
        - ISR 자산, 우주 위성, 사이버 작전 포함
        """

    @staticmethod
    def create_urban_warfare_scenario(seed=None) -> CombatKnowledgeGraph:
        """
        도시 전투 시나리오 (시가전)
        - 소대·중대급 근접전
        - 부수피해 최소화 제약 반영
        - 드론 정찰, 저격, 건물 제거 역학
        """
```

---

### STEP 8 — 현실 전투 역학 개선

**우선순위**: 중기~장기

#### 8-1. 탄약 소모 모델 (Ammunition Consumption)

```python
AMMO_CONSUMPTION_PER_ENGAGEMENT = {
    "infantry":           {"small_arms": 200, "atgm": 1, "grenade": 3},
    "armor":              {"main_gun_round": 5, "coax_ammo": 100},
    "artillery":          {"155mm_round": 40, "illumination": 5},
    "fighter":            {"aim9x": 2, "aim120": 4, "jdam": 4},
    "surface_combatant":  {"ssm": 8, "gun_round": 50, "sam": 4},
    "submarine":          {"torpedo": 2, "cruise_missile": 4},
}
```

#### 8-2. 전투피해평가(BDA) 정밀화

```python
@dataclass
class BattleDamageAssessment:
    """전투 피해 평가 (BDA)"""
    target_unit_id: str
    strike_type: str           # "direct_fire" | "indirect" | "air_strike" | "missile"
    pre_strike_strength: float # 타격 전 전투력
    post_strike_strength: float # 타격 후 전투력
    killed_in_action: int
    wounded_in_action: int
    equipment_destroyed: int
    equipment_damaged: int
    confidence_level: float    # BDA 신뢰도 [0, 1] (안개 전쟁 반영)
```

#### 8-3. 보급/군수 모델 (Logistics Model)

```python
@dataclass
class SupplyStatus:
    """부대 보급 상태"""
    unit_id: str
    ammo_days: float           # 잔여 탄약 (일수 기준)
    fuel_days: float           # 잔여 연료 (일수)
    food_days: float           # 잔여 식량 (일수)
    medical_capacity: float    # 의무 지원 능력 [0, 1]
    resupply_route_open: bool  # 보급선 개방 여부

    def combat_sustainability(self) -> float:
        """전투 지속 가능 일수 (최소값 기준)"""
        return min(self.ammo_days, self.fuel_days, self.food_days)
```

#### 8-4. 전자기스펙트럼(EMS) 관리

```python
@dataclass
class ElectromagneticEnvironment:
    """전자기 환경"""
    ew_intensity: float        # EW 강도 [0, 1]
    gps_degradation: float     # GPS 저하 수준 [0, 1]
    comms_jamming: float       # 통신 재밍 [0, 1]
    radar_detection_range: float  # 레이더 탐지 거리 (EW 영향)

    def apply_to_unit(self, unit: Unit) -> Unit:
        """EMS 효과를 유닛에 적용한 사본 반환"""
```

---

### STEP 9 — 국제법·교전규칙(ROE) 온톨로지

**우선순위**: 장기

#### 9-1. RulesOfEngagement 클래스

```python
@dataclass
class RulesOfEngagement:
    """교전규칙"""
    mission_type: str               # "peacekeeping" | "wartime" | "counterterrorism"

    # 무력 사용 기준
    force_threshold: str            # "imminent_threat" | "hostile_act" | "hostile_intent"
    collateral_damage_limit: float  # 부수피해 허용 한도 [0, 1]

    # 금지 사항
    no_strike_zones: List[str]      # 공격 금지 지역 (병원·학교 등)
    no_strike_unit_types: List[str] # 공격 금지 유닛 유형
    cyber_restrictions: List[str]   # 사이버 공격 제한

    # LOAC (Law of Armed Conflict) 검증
    def is_strike_lawful(self, target, collateral_damage: float) -> Tuple[bool, str]:
        """국제인도법 준수 여부 검증 → (합법성, 이유)"""
```

#### 9-2. EthicalConstraintChecker

```python
class EthicalConstraintChecker:
    """윤리적 제약 자동 검증기 (AI 의사결정 지원)"""

    def check_proportionality(self, military_advantage, collateral_damage) -> bool:
        """군사적 이익 대비 부수피해 비례성 검증"""

    def check_discrimination(self, target) -> bool:
        """전투원/민간인 구별 원칙 검증"""

    def check_military_necessity(self, action, alternatives) -> bool:
        """군사적 필요성 원칙 검증"""

    def generate_roe_report(self) -> str:
        """ROE 준수 보고서 생성"""
```

---

## 3. 구현 우선순위 및 일정 제안

| 단계 | 내용 | 예상 규모 | 우선순위 |
|------|------|-----------|---------|
| **STEP 1** | BranchType + EchelonType + Unit 확장 | 소 (50줄) | 즉시 |
| **STEP 2** | UnitType 확장 (30+ 유닛) | 중 (150줄) | 즉시 |
| **STEP 3** | Capability 확장 (20+ 필드) | 중 (200줄) | 즉시 |
| **STEP 4** | 현실 병력 규모 테이블 | 소 (100줄) | 즉시 |
| **STEP 5** | DomainType + Cyber/Space 효과 | 중 (200줄) | 단기 |
| **STEP 6** | C2 구조 + 합동작전 | 대 (300줄) | 단기 |
| **STEP 7** | ScenarioFactory 프리셋 | 대 (500줄) | 단기 |
| **STEP 8** | 탄약/BDA/군수/EMS 모델 | 대 (400줄) | 중기 |
| **STEP 9** | ROE + 윤리 검증기 | 중 (300줄) | 장기 |

---

## 4. 파급 효과 분석

### 4-1. GNN 모델에 미치는 영향

| 변경 사항 | GNN 영향 |
|-----------|---------|
| UnitType 8→40+ 종 | 노드 특성 벡터 차원 증가 (현재 30D → ~60D) |
| Branch + Echelon 추가 | 새로운 노드 속성 2개 (원-핫) |
| Capability 8→20+ 필드 | capability.to_vector() 차원 증가 |
| 지휘 계층 관계 | 새로운 엣지 유형 (COMMANDS, SUBORDINATE_OF) |
| 도메인 효과 | 도메인별 별도 그래프 서브네트워크 |

**권장 대응**: `BayesianHGT` 에서 `node_in_dim=32` → `node_in_dim=64` 상향 조정

### 4-2. RL 에이전트에 미치는 영향

| 변경 사항 | RL 영향 |
|-----------|---------|
| 유닛 종류 증가 | 행동 공간 확장 필요 (현재 6종 → 20+ 종 행동) |
| 군종별 행동 차이 | 군종별 별도 Actor-Critic 네트워크 고려 |
| 지휘 계층 | 계층적 행동 공간 (HRL: 사단→연대→대대 순차) |
| 합동작전 | MAPPO 확장 (군종별 에이전트) |

### 4-3. 시뮬레이터에 미치는 영향

| 변경 사항 | 시뮬레이터 영향 |
|-----------|----------------|
| 해군 전력 | 해상 전투 역학 (`naval_combat_engine.py` 신설) |
| 미사일 | 탄도 궤적·요격 계산 모듈 |
| 잠수함 | 수중 탐지 모델 (소나 방정식) |
| 사이버 | 사이버 공격 효과 전파 모델 |
| 우주 자산 | ISR 커버리지 계산 |

---

## 5. 즉시 착수 권장 사항 (Quick Wins)

다음 3가지는 기존 코드 변경 없이 파일 추가만으로 가능한 **즉시 구현 항목**이다.

### Q1. `ontology/military_units.py` 신설
현실 군종/계층/병력 규모 상수 테이블 정의 (코드 영향 없음).

### Q2. `ontology/scenario_presets.py` 신설
ScenarioFactory 래핑으로 6종 현실 시나리오 프리셋 제공.

### Q3. `configs/scenarios/` 디렉토리 신설
YAML 형태의 시나리오 설정 파일 (한반도 방어, 상륙작전, 공중전 등).

---

## 6. 창의적 확장 아이디어

### 6-1. 정보 온톨로지 (Intelligence Ontology)

```
INTELReport: HUMINT·SIGINT·IMINT·MASINT 정보 보고서
             → 불확실성 감소 효과, GNN 입력으로 활용
```

### 6-2. 심리전·영향력 작전

```
InfluenceOperation: 적 사기·민간 지지·동맹 결속 영향 모델링
                    → 보상 함수에 "비물리적 효과" 추가 가능
```

### 6-3. 기후/환경 요소

```
WeatherEffect: 기온·강수·가시거리·바람 → 전투 효과 수정자
SeasonEffect:  동계·하계 작전 능력 차이 모델링
```

### 6-4. 연합작전 온톨로지

```
AllianceStructure: 한미연합사 수준의 지휘권 공유 모델
Coalition:         NATO 수준의 상호운용성(Interoperability) 지수
```

### 6-5. 하이브리드전 요소

```
HybridThreat: 정규전+비정규전+사이버+정보전 혼합 위협
GrayZoneOp:   분쟁 임계 이하 회색지대 작전 (군사·비군사 경계)
```

---

## 7. 단계별 파일 신설/수정 목록

| 파일 | 유형 | 내용 |
|------|------|------|
| `ontology/combat_schema.py` | 수정 | Unit + BranchType + EchelonType + 확장 UnitType + Capability |
| `ontology/military_units.py` | **신설** | 군종·계층·현실 병력 규모 상수 테이블 |
| `ontology/scenario_presets.py` | **신설** | 6종 현실 시나리오 팩토리 |
| `ontology/joint_operations.py` | **신설** | C2 구조·합동화력·연합작전 |
| `ontology/multidomain.py` | 수정 | Cyber/Space/Subsurface 도메인 추가 |
| `ontology/roe_ethics.py` | **신설** | ROE·국제인도법 검증기 |
| `simulator/naval_engine.py` | **신설** | 해상 전투 역학 엔진 |
| `simulator/missile_model.py` | **신설** | 탄도·순항·대함 미사일 모델 |
| `simulator/cyber_effects.py` | **신설** | 사이버 공격 효과 전파 |
| `configs/scenarios/*.yaml` | **신설** | 현실 시나리오 YAML 설정 |

---

*본 보고서는 FALCON 온톨로지 발전의 기술 명세서로 활용하며,
구현 우선순위는 프로젝트 일정 및 연구 방향에 따라 조정한다.*
