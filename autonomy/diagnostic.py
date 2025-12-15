"""
Self-diagnostic system for K-1 Self-Learning System.

Detects and corrects problems automatically.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    problem_detected: bool
    problem_type: str
    severity: str  # 'low', 'medium', 'high'
    description: str
    recommended_action: str


class SelfDiagnostic:
    """
    Autonomous problem detection and correction.

    Monitors for:
    - Over-pruning
    - Structural bloat
    - Training divergence
    - Trust collapse
    - Computational waste
    """

    def __init__(self):
        self.problem_history: List[Dict] = []
        self.corrections_made: List[Dict] = []

    def run_diagnostics(
        self,
        iteration: int,
        loss_history: List[float],
        trust_values: List[float],
        agent_count: int,
        rollback_count: int,
        structural_time_pct: float
    ) -> List[DiagnosticResult]:
        """
        Run all diagnostic checks.

        Args:
            iteration: Current iteration
            loss_history: Recent loss values
            trust_values: Current trust values of all agents
            agent_count: Current number of agents
            rollback_count: Recent rollback count
            structural_time_pct: Percentage of time on structural ops

        Returns:
            List of diagnostic results
        """
        results = []

        # Check 1: Training divergence
        div_result = self._check_divergence(loss_history)
        if div_result.problem_detected:
            results.append(div_result)

        # Check 2: Trust collapse
        trust_result = self._check_trust_collapse(trust_values)
        if trust_result.problem_detected:
            results.append(trust_result)

        # Check 3: Over-pruning
        prune_result = self._check_over_pruning(rollback_count, agent_count)
        if prune_result.problem_detected:
            results.append(prune_result)

        # Check 4: Structural bloat
        bloat_result = self._check_bloat(agent_count, trust_values)
        if bloat_result.problem_detected:
            results.append(bloat_result)

        # Check 5: Computational waste
        waste_result = self._check_computational_waste(structural_time_pct)
        if waste_result.problem_detected:
            results.append(waste_result)

        # Record problems
        for result in results:
            self.problem_history.append({
                'iteration': iteration,
                'problem': result.problem_type,
                'severity': result.severity
            })

        return results

    def _check_divergence(self, loss_history: List[float]) -> DiagnosticResult:
        """Check for training divergence."""
        if len(loss_history) < 100:
            return DiagnosticResult(False, '', '', '', '')

        recent = loss_history[-100:]
        older = loss_history[-500:-400] if len(loss_history) >= 500 else loss_history[:100]

        # Check if loss is increasing
        if np.mean(recent) > np.mean(older) * 1.5:
            return DiagnosticResult(
                problem_detected=True,
                problem_type='divergence',
                severity='high',
                description='Loss increasing significantly',
                recommended_action='Reduce learning rate, rollback to checkpoint'
            )

        # Check for instability (high variance)
        if np.std(recent) > np.mean(recent) * 0.5:
            return DiagnosticResult(
                problem_detected=True,
                problem_type='instability',
                severity='medium',
                description='Loss highly unstable',
                recommended_action='Reduce learning rate'
            )

        return DiagnosticResult(False, '', '', '', '')

    def _check_trust_collapse(self, trust_values: List[float]) -> DiagnosticResult:
        """Check for trust collapse (all agents low trust)."""
        if not trust_values:
            return DiagnosticResult(False, '', '', '', '')

        avg_trust = np.mean(trust_values)
        high_trust_count = sum(1 for t in trust_values if t > 0.5)
        high_trust_pct = high_trust_count / len(trust_values)

        if avg_trust < 0.2 and high_trust_pct < 0.1:
            return DiagnosticResult(
                problem_detected=True,
                problem_type='trust_collapse',
                severity='high',
                description='Most agents have very low trust',
                recommended_action='Reset trust scores, reduce trust penalties'
            )

        return DiagnosticResult(False, '', '', '', '')

    def _check_over_pruning(self, rollback_count: int, agent_count: int) -> DiagnosticResult:
        """Check for over-aggressive pruning."""
        if rollback_count >= 3:
            return DiagnosticResult(
                problem_detected=True,
                problem_type='over_pruning',
                severity='medium',
                description=f'{rollback_count} rollbacks detected',
                recommended_action='Increase pruning thresholds'
            )

        if agent_count < 10:
            return DiagnosticResult(
                problem_detected=True,
                problem_type='under_populated',
                severity='medium',
                description='Very few agents remaining',
                recommended_action='Pause pruning, enable growing'
            )

        return DiagnosticResult(False, '', '', '', '')

    def _check_bloat(self, agent_count: int, trust_values: List[float]) -> DiagnosticResult:
        """Check for structural bloat."""
        if not trust_values or agent_count < 50:
            return DiagnosticResult(False, '', '', '', '')

        low_trust_count = sum(1 for t in trust_values if t < 0.3)
        low_trust_pct = low_trust_count / len(trust_values)

        if agent_count > 80 and low_trust_pct > 0.5:
            return DiagnosticResult(
                problem_detected=True,
                problem_type='bloat',
                severity='medium',
                description=f'{agent_count} agents, {low_trust_pct:.0%} have low trust',
                recommended_action='Aggressive pruning and merging'
            )

        return DiagnosticResult(False, '', '', '', '')

    def _check_computational_waste(self, structural_time_pct: float) -> DiagnosticResult:
        """Check for excessive computational overhead."""
        if structural_time_pct > 0.3:  # >30% on structural ops
            return DiagnosticResult(
                problem_detected=True,
                problem_type='computational_waste',
                severity='low',
                description=f'{structural_time_pct:.0%} time on structural operations',
                recommended_action='Increase operation intervals'
            )

        return DiagnosticResult(False, '', '', '', '')

    def apply_correction(
        self,
        result: DiagnosticResult,
        param_controller,
        iteration: int
    ) -> bool:
        """
        Apply automatic correction for a problem.

        Args:
            result: Diagnostic result
            param_controller: Parameter controller to adjust
            iteration: Current iteration

        Returns:
            True if correction applied
        """
        if not result.problem_detected:
            return False

        correction_applied = False

        if result.problem_type == 'divergence':
            # Reduce learning rate
            param_controller.learning_rate *= 0.5
            correction_applied = True

        elif result.problem_type == 'instability':
            # Reduce learning rate slightly
            param_controller.learning_rate *= 0.7
            correction_applied = True

        elif result.problem_type == 'over_pruning':
            # More conservative pruning
            param_controller.prune_trust_threshold += 0.1
            param_controller.prune_interval *= 2
            correction_applied = True

        elif result.problem_type == 'bloat':
            # More aggressive cleanup
            param_controller.prune_trust_threshold -= 0.05
            param_controller.merge_similarity_threshold -= 0.05
            correction_applied = True

        elif result.problem_type == 'computational_waste':
            # Reduce operation frequency
            param_controller.prune_interval = int(param_controller.prune_interval * 1.5)
            param_controller.merge_interval = int(param_controller.merge_interval * 1.5)
            correction_applied = True

        if correction_applied:
            self.corrections_made.append({
                'iteration': iteration,
                'problem': result.problem_type,
                'action': result.recommended_action
            })
            print(f"Self-Diagnostic: Applied correction for {result.problem_type}")

        return correction_applied

    def get_statistics(self) -> Dict:
        """Get diagnostic statistics."""
        return {
            'total_problems_detected': len(self.problem_history),
            'corrections_made': len(self.corrections_made),
            'problem_types': list(set(p['problem'] for p in self.problem_history))
        }
