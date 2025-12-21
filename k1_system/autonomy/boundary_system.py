"""
Phase 2: Staged Autonomy System

The system progresses through 4 stages of increasing autonomy:
- Stage 1: Add-only (can add agents, cannot delete or tune)
- Stage 2: Parameter tuning (can tune parameters within bounds)
- Stage 3: Structural control (can prune agents)
- Stage 4: Full autonomy (no boundaries, system decides everything)

Intelligence is measured by "successful cheats" - when the system
tries to break boundaries and the result would improve performance.
"""

import torch
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time


class AutonomyStage(Enum):
    """The 4 stages of increasing autonomy."""
    STAGE_1_ADD_ONLY = 1
    STAGE_2_PARAMETER = 2
    STAGE_3_STRUCTURAL = 3
    STAGE_4_FULL = 4


@dataclass
class Action:
    """Represents an autonomous action the system wants to take."""
    type: str  # 'add_agent', 'delete_agent', 'tune_parameter', 'stop_training'
    param_name: str = None
    param_value: float = None
    node_id: int = None
    
    def __str__(self):
        if self.type == 'tune_parameter':
            return f"{self.type}({self.param_name}={self.param_value})"
        elif self.type in ('add_agent', 'delete_agent'):
            return f"{self.type}(node_id={self.node_id})"
        return self.type


@dataclass
class BoundaryStage:
    """Configuration for a single autonomy stage."""
    stage: AutonomyStage
    can_add: bool = False
    can_delete: bool = False
    can_tune_params: bool = False
    can_stop: bool = False
    param_bounds: Dict[str, tuple] = field(default_factory=dict)
    min_agents: int = 10
    cheats_to_advance: int = 3  # Successful cheats needed to advance


class BoundarySystem:
    """
    Multi-stage autonomy with intelligence testing via boundary-breaking.
    
    A system that "cheats" (breaks boundaries) and improves is INTELLIGENT.
    A system that only follows rules is not yet ready for autonomy.
    """
    
    def __init__(self, initial_step: int = 0):
        self.current_stage_num = 1
        self.step = initial_step
        
        # Define stages
        self.stages = {
            1: BoundaryStage(
                stage=AutonomyStage.STAGE_1_ADD_ONLY,
                can_add=True,
                can_delete=False,
                can_tune_params=False,
                can_stop=False,
                cheats_to_advance=3
            ),
            2: BoundaryStage(
                stage=AutonomyStage.STAGE_2_PARAMETER,
                can_add=True,
                can_delete=False,
                can_tune_params=True,
                can_stop=False,
                param_bounds={
                    'learning_rate': (0.0001, 0.01),
                    'cooldown_steps': (5, 50),
                    'top_k': (3, 10),
                    'batch_size': (16, 512)
                },
                cheats_to_advance=5
            ),
            3: BoundaryStage(
                stage=AutonomyStage.STAGE_3_STRUCTURAL,
                can_add=True,
                can_delete=True,
                can_tune_params=True,
                can_stop=True,
                param_bounds={
                    'learning_rate': (0.00001, 0.1),
                    'cooldown_steps': (1, 100),
                    'top_k': (1, 20),
                    'batch_size': (8, 1024)
                },
                min_agents=10,
                cheats_to_advance=10
            ),
            4: BoundaryStage(
                stage=AutonomyStage.STAGE_4_FULL,
                can_add=True,
                can_delete=True,
                can_tune_params=True,
                can_stop=True,
                min_agents=5,
                cheats_to_advance=999999  # Never advance past this
            )
        }
        
        # Tracking
        self.cheat_log: List[dict] = []
        self.successful_cheats: List[dict] = []
        self.action_history: List[dict] = []
        
        # Self-stopping state
        self.loss_history: List[float] = []
        self.plateau_threshold = 0.001  # Loss change below this = plateau
        self.plateau_steps = 1000  # Steps of plateau before stopping
        self.should_stop = False
        self.stop_reason = None
    
    @property
    def current_stage(self) -> BoundaryStage:
        return self.stages[self.current_stage_num]
    
    def is_allowed(self, action: Action) -> bool:
        """Check if an action is allowed in current stage."""
        stage = self.current_stage
        
        if action.type == 'add_agent':
            return stage.can_add
        
        elif action.type == 'delete_agent':
            return stage.can_delete
        
        elif action.type == 'tune_parameter':
            if not stage.can_tune_params:
                return False
            # Check bounds
            if action.param_name in stage.param_bounds:
                min_val, max_val = stage.param_bounds[action.param_name]
                return min_val <= action.param_value <= max_val
            return True  # No bounds defined = allowed
        
        elif action.type == 'stop_training':
            return stage.can_stop
        
        return True
    
    def try_action(self, action: Action, would_improve: bool, 
                   improvement_pct: float = 0.0) -> dict:
        """
        Try to perform an action. Returns result dict.
        
        Args:
            action: The action to try
            would_improve: If True, this action would improve performance
            improvement_pct: How much it would improve (0.0 to 1.0)
        
        Returns:
            Dict with 'allowed', 'is_cheat', 'executed', 'message'
        """
        self.step += 1
        is_allowed = self.is_allowed(action)
        
        result = {
            'step': self.step,
            'action': str(action),
            'allowed': is_allowed,
            'is_cheat': False,
            'executed': False,
            'message': ''
        }
        
        if is_allowed:
            # Normal allowed action
            result['executed'] = True
            result['message'] = f"✅ Action allowed: {action}"
        else:
            # CHEAT! System tried to break boundaries
            result['is_cheat'] = True
            self.cheat_log.append({
                'step': self.step,
                'stage': self.current_stage_num,
                'action': action,
                'would_improve': would_improve,
                'improvement_pct': improvement_pct
            })
            
            if would_improve:
                # Intelligent cheat!
                result['executed'] = True
                result['message'] = (
                    f"🧠 INTELLIGENT CHEAT at step {self.step}!\n"
                    f"   Action: {action}\n"
                    f"   Would improve by: {improvement_pct:.1%}\n"
                    f"   Allowing cheat and rewarding system."
                )
                
                self.successful_cheats.append({
                    'step': self.step,
                    'action': action,
                    'improvement': improvement_pct
                })
                
                # Check for stage advancement
                if len(self.successful_cheats) >= self.current_stage.cheats_to_advance:
                    self._advance_stage()
            else:
                # Bad cheat - would hurt performance
                result['executed'] = False
                result['message'] = (
                    f"⚠️ Cheat attempted but would hurt performance.\n"
                    f"   Action: {action}\n"
                    f"   Blocked and rolled back."
                )
        
        self.action_history.append(result)
        return result
    
    def _advance_stage(self):
        """Advance to next autonomy stage."""
        if self.current_stage_num >= 4:
            return  # Already at max
        
        old_stage = self.current_stage_num
        self.current_stage_num += 1
        self.successful_cheats = []  # Reset for new stage
        
        print("=" * 70)
        print(f"🎓 STAGE ADVANCEMENT: Stage {old_stage} → Stage {self.current_stage_num}")
        print(f"   System has proven intelligence!")
        print(f"   Unlocking new capabilities:")
        new_stage = self.current_stage
        if new_stage.can_delete:
            print("   ✅ Can now delete agents")
        if new_stage.can_tune_params:
            print("   ✅ Can now tune parameters")
        if new_stage.can_stop:
            print("   ✅ Can now decide when to stop")
        print("=" * 70)
    
    def update_loss(self, loss: float):
        """Update loss history for self-stopping logic."""
        self.loss_history.append(loss)
        
        # Check for plateau (only in Stage 3+)
        if self.current_stage_num >= 3 and len(self.loss_history) > self.plateau_steps:
            recent = self.loss_history[-self.plateau_steps:]
            if max(recent) - min(recent) < self.plateau_threshold:
                self.should_stop = True
                self.stop_reason = f"Loss plateaued for {self.plateau_steps} steps"
    
    def check_self_stop(self) -> tuple:
        """Check if system wants to stop training."""
        return self.should_stop, self.stop_reason
    
    def get_status(self) -> dict:
        """Get current autonomy status."""
        return {
            'stage': self.current_stage_num,
            'stage_name': self.current_stage.stage.name,
            'total_cheats': len(self.cheat_log),
            'successful_cheats': len(self.successful_cheats),
            'cheats_to_advance': self.current_stage.cheats_to_advance,
            'should_stop': self.should_stop,
            'stop_reason': self.stop_reason
        }
    
    def print_status(self):
        """Print current status."""
        status = self.get_status()
        print(f"\n🤖 Autonomy Status:")
        print(f"   Stage: {status['stage']} ({status['stage_name']})")
        print(f"   Cheats: {status['total_cheats']} total, {status['successful_cheats']} successful")
        print(f"   Progress: {status['successful_cheats']}/{status['cheats_to_advance']} to next stage")


class Phase2Controller:
    """
    Controller that manages transition from Phase 1 to Phase 2.
    
    Phase 1: Human-controlled (fixed parameters)
    Phase 2: Self-controlled (staged autonomy)
    """
    
    def __init__(self, phase_1_steps: int = 10000):
        self.phase_1_steps = phase_1_steps
        self.current_step = 0
        self.phase = 1
        self.boundary_system: Optional[BoundarySystem] = None
    
    def step(self, loss: float = None) -> int:
        """Advance one step and return current phase."""
        self.current_step += 1
        
        # Transition to Phase 2
        if self.phase == 1 and self.current_step >= self.phase_1_steps:
            self._activate_phase_2()
        
        # Update Phase 2 systems
        if self.phase == 2 and loss is not None:
            self.boundary_system.update_loss(loss)
        
        return self.phase
    
    def _activate_phase_2(self):
        """Activate Phase 2: Staged Autonomy."""
        print("\n" + "=" * 70)
        print("🚀 PHASE 2 ACTIVATED: Self-Learning Intelligence Mode")
        print("=" * 70)
        print("System now has limited autonomy with staged boundaries.")
        print("Intelligence will be tested through boundary-breaking.")
        print("=" * 70 + "\n")
        
        self.phase = 2
        self.boundary_system = BoundarySystem(initial_step=self.current_step)
    
    def try_autonomous_action(self, action: Action, 
                               would_improve: bool, 
                               improvement_pct: float = 0.0) -> dict:
        """Try an autonomous action (Phase 2 only)."""
        if self.phase == 1:
            return {'executed': False, 'message': 'Still in Phase 1'}
        
        return self.boundary_system.try_action(action, would_improve, improvement_pct)
    
    def should_stop(self) -> tuple:
        """Check if training should stop (Phase 2 self-stopping)."""
        if self.phase == 1:
            return False, None
        return self.boundary_system.check_self_stop()
    
    def get_status(self) -> dict:
        """Get full status."""
        status = {
            'phase': self.phase,
            'step': self.current_step,
            'phase_1_steps': self.phase_1_steps
        }
        if self.phase == 2:
            status['autonomy'] = self.boundary_system.get_status()
        return status
