"""
maneuver_engine.py — A. 지형 기반 기동 엔진
유닛 위치 이동 / A* 경로 / LOS / 측면·포위 판정
학술 연구용 합성 데이터
"""
from __future__ import annotations
import heapq, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from ontology.combat_schema import (
    Unit, UnitType, UnitStatus, ForceAlignment,
    TerrainType, Position, CombatKnowledgeGraph
)

UNIT_ENGAGEMENT_RANGE: Dict[str, float] = {
    "infantry": 3.0, "armor": 5.0, "artillery": 15.0,
    "aviation": 12.0, "engineer": 2.0, "signal": 0.5,
    "logistics": 0.5, "electronic_warfare": 8.0,
}
UNIT_MOVE_SPEED: Dict[str, float] = {
    "infantry": 1.5, "armor": 3.5, "artillery": 1.0,
    "aviation": 10.0, "engineer": 1.2, "signal": 2.0,
    "logistics": 2.5, "electronic_warfare": 2.0,
}

def euclidean(ax,ay,bx,by): return math.sqrt((ax-bx)**2+(ay-by)**2)

@dataclass
class TerrainCell:
    x: int; y: int; terrain: TerrainType
    elevation: float; is_obstacle: bool = False

    @property
    def movement_cost(self) -> float:
        base = {TerrainType.OPEN:1.0, TerrainType.FOREST:1.8,
                TerrainType.URBAN:1.5, TerrainType.MOUNTAIN:2.5,
                TerrainType.RIVER:3.0, TerrainType.COASTAL:1.3}.get(self.terrain,1.0)
        return base + self.elevation*0.5

    @property
    def concealment_bonus(self) -> float:
        return {TerrainType.OPEN:0.0, TerrainType.FOREST:0.6,
                TerrainType.URBAN:0.4, TerrainType.MOUNTAIN:0.3,
                TerrainType.RIVER:0.1, TerrainType.COASTAL:0.2}.get(self.terrain,0.0)

    @property
    def defense_bonus(self) -> float:
        return {TerrainType.OPEN:0.0, TerrainType.FOREST:0.3,
                TerrainType.URBAN:0.5, TerrainType.MOUNTAIN:0.6,
                TerrainType.RIVER:0.2, TerrainType.COASTAL:0.2}.get(self.terrain,0.0)


class TerrainMap:
    def __init__(self, width:int=30, height:int=30, seed:int=0):
        self.width=width; self.height=height
        self.cells: Dict[Tuple[int,int],TerrainCell]={}
        rng=np.random.RandomState(seed)
        zones=[(0,10,0,30,TerrainType.OPEN),(10,20,0,30,TerrainType.FOREST),
               (20,30,0,15,TerrainType.MOUNTAIN),(20,30,15,30,TerrainType.OPEN),
               (12,18,10,20,TerrainType.URBAN),(5,8,0,30,TerrainType.RIVER)]
        for x in range(width):
            for y in range(height):
                t=TerrainType.OPEN
                for x0,x1,y0,y1,tt in zones:
                    if x0<=x<x1 and y0<=y<y1: t=tt
                e=float(rng.uniform(0,0.4))
                if t==TerrainType.MOUNTAIN: e+=0.4
                obs=t==TerrainType.RIVER and rng.random()<0.3
                self.cells[(x,y)]=TerrainCell(x,y,t,e,obs)

    def get(self,x,y): return self.cells.get((x,y))
    def in_bounds(self,x,y): return 0<=x<self.width and 0<=y<self.height
    def neighbors(self,x,y):
        dirs=[(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        return [(x+dx,y+dy) for dx,dy in dirs
                if self.in_bounds(x+dx,y+dy)
                and not (self.cells.get((x+dx,y+dy)) or TerrainCell(0,0,TerrainType.OPEN,0,False)).is_obstacle]


def check_los(tmap, ax,ay, bx,by, max_range) -> bool:
    if euclidean(ax,ay,bx,by)>max_range: return False
    x0,y0=int(round(ax)),int(round(ay)); x1,y1=int(round(bx)),int(round(by))
    dx,dy=abs(x1-x0),abs(y1-y0)
    sx=1 if x0<x1 else -1; sy=1 if y0<y1 else -1
    err=dx-dy; cx,cy=x0,y0; steps=0
    while (cx,cy)!=(x1,y1) and steps<100:
        cell=tmap.get(cx,cy)
        if cell and cell.is_obstacle: return False
        if cell and cell.terrain==TerrainType.MOUNTAIN and steps>2: return False
        e2=2*err
        if e2>-dy: err-=dy; cx+=sx
        if e2<dx:  err+=dx; cy+=sy
        steps+=1
    return True


def astar(tmap, start, goal, unit_type="infantry"):
    if start==goal: return [start]
    def h(p): return euclidean(p[0],p[1],goal[0],goal[1])
    heap=[(h(start),0.0,start,[start])]; visited=set()
    while heap:
        _,cost,cur,path=heapq.heappop(heap)
        if cur in visited: continue
        visited.add(cur)
        if cur==goal: return path
        for nx,ny in tmap.neighbors(*cur):
            if (nx,ny) in visited: continue
            cell=tmap.get(nx,ny)
            nc=cost+(cell.movement_cost if cell else 1.0)
            heapq.heappush(heap,(nc+h((nx,ny)),nc,(nx,ny),path+[(nx,ny)]))
    return None


@dataclass
class ManeuverAssessment:
    unit_id: str
    can_flank: bool; can_envelop: bool
    flank_angle: float; envelopment_coverage: float
    cover_score: float; in_range_enemies: List[str]
    recommended_action: str

    def to_dict(self):
        return {"unit_id":self.unit_id,"can_flank":self.can_flank,
                "can_envelop":self.can_envelop,"flank_angle":round(self.flank_angle,1),
                "cover_score":round(self.cover_score,2),
                "in_range_enemies":len(self.in_range_enemies),
                "recommended_action":self.recommended_action}


class ManeuverEngine:
    def __init__(self, map_size:int=30, seed:int=0):
        self.terrain_map=TerrainMap(map_size,map_size,seed)
        self.rng=np.random.RandomState(seed)
        self.move_log: List[Dict]=[]

    def move_unit(self, unit:Unit, tx:float, ty:float, dt:float=1.0) -> Tuple[float,float,str]:
        if unit.status == UnitStatus.DESTROYED:
            return unit.position.x, unit.position.y, "immobile"
        speed=UNIT_MOVE_SPEED.get(unit.unit_type.value,1.5)
        fuel_pen=max(0.1,unit.capability.fuel_level)
        eff_speed=speed*fuel_pen*dt
        cx,cy=int(round(unit.position.x)),int(round(unit.position.y))
        cell=self.terrain_map.get(cx,cy)
        if cell: eff_speed/=cell.movement_cost
        dx=tx-unit.position.x; dy=ty-unit.position.y
        dist=math.sqrt(dx**2+dy**2)
        if dist<0.1: return unit.position.x,unit.position.y,"arrived"
        mv=min(eff_speed,dist)
        nx=float(np.clip(unit.position.x+(dx/dist)*mv,0,self.terrain_map.width-1))
        ny=float(np.clip(unit.position.y+(dy/dist)*mv,0,self.terrain_map.height-1))
        self.move_log.append({"unit_id":unit.unit_id,"from":(unit.position.x,unit.position.y),"to":(nx,ny)})
        unit.position.x=nx; unit.position.y=ny
        return nx,ny,"moving"

    def plan_route(self, unit:Unit, gx:float, gy:float):
        s=(int(round(unit.position.x)),int(round(unit.position.y)))
        g=(int(round(gx)),int(round(gy)))
        return astar(self.terrain_map,s,g,unit.unit_type.value)

    def can_engage(self, a:Unit, d:Unit) -> bool:
        if a.status==UnitStatus.DESTROYED or d.status==UnitStatus.DESTROYED: return False
        mr=UNIT_ENGAGEMENT_RANGE.get(a.unit_type.value,3.0)
        return check_los(self.terrain_map,a.position.x,a.position.y,
                         d.position.x,d.position.y,mr)

    def find_engageable_targets(self, unit:Unit, kg:CombatKnowledgeGraph) -> List[Unit]:
        ea=ForceAlignment.RED if unit.alignment==ForceAlignment.BLUE else ForceAlignment.BLUE
        return [u for u in kg.units.values() if u.alignment==ea and self.can_engage(unit,u)]

    def assess_maneuver(self, unit:Unit, kg:CombatKnowledgeGraph) -> ManeuverAssessment:
        ea=ForceAlignment.RED if unit.alignment==ForceAlignment.BLUE else ForceAlignment.BLUE
        enemies=[u for u in kg.units.values() if u.alignment==ea and u.status!=UnitStatus.DESTROYED]
        in_range=[u for u in enemies if self.can_engage(unit,u)]
        cx,cy=int(round(unit.position.x)),int(round(unit.position.y))
        cell=self.terrain_map.get(cx,cy)
        cover=cell.concealment_bonus if cell else 0.0

        flank_angle=0.0; can_flank=False
        if enemies:
            cl=min(enemies,key=lambda e:euclidean(unit.position.x,unit.position.y,e.position.x,e.position.y))
            tf=math.atan2(5.0-cl.position.y,5.0-cl.position.x)
            tu=math.atan2(unit.position.y-cl.position.y,unit.position.x-cl.position.x)
            flank_angle=abs(math.degrees(tu-tf))%180
            can_flank=flank_angle>45

        coverage=0.0; can_envelop=False
        if enemies:
            cl2=min(enemies,key=lambda e:euclidean(unit.position.x,unit.position.y,e.position.x,e.position.y))
            fa=unit.alignment
            friendlies=[u for u in kg.units.values() if u.alignment==fa and u.status!=UnitStatus.DESTROYED]
            angles=sorted([math.degrees(math.atan2(f.position.y-cl2.position.y,f.position.x-cl2.position.x)) for f in friendlies])
            if len(angles)>=2:
                gaps=[angles[i+1]-angles[i] for i in range(len(angles)-1)]
                gaps.append(360-angles[-1]+angles[0])
                coverage=max(0,(360-max(gaps))/360)
            can_envelop=coverage>0.6

        if can_envelop and in_range: action="포위_공격"
        elif can_flank and in_range: action="측면_공격"
        elif in_range: action="정면_교전"
        elif cover>0.4: action="은폐_대기"
        else: action="전진"

        return ManeuverAssessment(unit.unit_id,can_flank,can_envelop,
                                   flank_angle,coverage,cover,
                                   [u.unit_id for u in in_range],action)

    def run_maneuver_step(self, kg:CombatKnowledgeGraph,
                          blue_targets:Optional[Dict[str,Tuple[float,float]]]=None,
                          dt:float=1.0) -> Dict:
        moved=0; engageable=[]
        blue=[u for u in kg.units.values() if u.alignment==ForceAlignment.BLUE and u.status!=UnitStatus.DESTROYED]
        red =[u for u in kg.units.values() if u.alignment==ForceAlignment.RED  and u.status!=UnitStatus.DESTROYED]
        for u in blue:
            tx,ty=(blue_targets.get(u.unit_id,(self.terrain_map.width*0.6,self.terrain_map.height*0.5))
                   if blue_targets else (self.terrain_map.width*0.6,self.terrain_map.height*0.5))
            self.move_unit(u,tx,ty,dt); moved+=1
        for u in red:
            if blue:
                cl=min(blue,key=lambda b:euclidean(u.position.x,u.position.y,b.position.x,b.position.y))
                self.move_unit(u,cl.position.x,cl.position.y,dt*0.7)
            moved+=1
        for bu in blue:
            for ru in red:
                if self.can_engage(bu,ru): engageable.append((bu.unit_id,ru.unit_id))
        assessments=[self.assess_maneuver(u,kg) for u in blue[:3]]
        return {"units_moved":moved,"engageable_pairs":engageable,
                "n_engageable":len(engageable),
                "maneuver_assessments":[a.to_dict() for a in assessments],
                "flanking_units":sum(1 for a in assessments if a.can_flank),
                "enveloping":any(a.can_envelop for a in assessments)}

    def get_spatial_stats(self, kg:CombatKnowledgeGraph) -> Dict:
        blue=[u for u in kg.units.values() if u.alignment==ForceAlignment.BLUE and u.status!=UnitStatus.DESTROYED]
        red =[u for u in kg.units.values() if u.alignment==ForceAlignment.RED  and u.status!=UnitStatus.DESTROYED]
        if not blue or not red: return {"front_distance":0,"avg_engagement_dist":0,"blue_centroid":(0,0),"red_centroid":(0,0),"n_engagement_pairs":0}
        bcx=float(np.mean([u.position.x for u in blue])); bcy=float(np.mean([u.position.y for u in blue]))
        rcx=float(np.mean([u.position.x for u in red]));  rcy=float(np.mean([u.position.y for u in red]))
        eds=[euclidean(bu.position.x,bu.position.y,ru.position.x,ru.position.y)
             for bu in blue for ru in red if euclidean(bu.position.x,bu.position.y,ru.position.x,ru.position.y)<10]
        return {"front_distance":round(euclidean(bcx,bcy,rcx,rcy),2),
                "avg_engagement_dist":round(float(np.mean(eds)) if eds else 0,2),
                "blue_centroid":(round(bcx,1),round(bcy,1)),
                "red_centroid":(round(rcx,1),round(rcy,1)),
                "n_engagement_pairs":len(eds)}


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory
    kg=ScenarioFactory.create_standard_scenario(n_blue=6,n_red=5,seed=42)
    engine=ManeuverEngine(map_size=30,seed=42)
    print("A. 기동 엔진 테스트")
    for step in range(5):
        r=engine.run_maneuver_step(kg)
        s=engine.get_spatial_stats(kg)
        print(f"  스텝{step+1}: 교전가능={r['n_engageable']}쌍 전선거리={s['front_distance']:.1f}km")
    bu=next(u for u in kg.units.values() if u.alignment==ForceAlignment.BLUE)
    route=engine.plan_route(bu,20.0,15.0)
    print(f"  A*경로: {len(route) if route else 0}웨이포인트")
    print("✅ 완료")
