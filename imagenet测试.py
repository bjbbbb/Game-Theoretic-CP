#!/usr/bin/env python
import argparse
import os
import random
import numpy as np
import torch
import logging
from tqdm import tqdm
import pandas as pd
import gc
import nashpy as nash

# --- FIX: Import the missing function ---
from torchcp.utils.common import calculate_conformal_value

# --- END FIX ---

# ------------------------------
# Setup Environment and Logging
# ------------------------------
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logging.basicConfig(
    filename='multi_cp_game_theory.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------
# Data Loading & Splitting
# ------------------------------
def load_model_predictions(adv_predictions_path, dataset_name, model_name, attack_name):
    """Load pre-computed model predictions (probabilities and labels)."""
    pred_file = os.path.join(adv_predictions_path, dataset_name.lower(), model_name, f"{attack_name}.pt")
    if not os.path.exists(pred_file):
        return None, None
    try:
        data = torch.load(pred_file, map_location='cpu')  # Load to CPU to manage memory
        return data['probabilities'], data['labels']
    except Exception as e:
        print(f"Error loading {pred_file}: {e}")
        return None, None


def split_dataset_three_way(total_size, cal_ratio=0.3, eval_ratio=0.3, seed=42):
    """Split dataset indices into calibration, evaluation, and test sets."""
    set_seed(seed)
    indices = torch.randperm(total_size)
    cal_size = int(total_size * cal_ratio)
    eval_size = int(total_size * eval_ratio)
    cal_indices = indices[:cal_size]
    eval_indices = indices[cal_size:cal_size + eval_size]
    test_indices = indices[cal_size + eval_size:]
    return cal_indices, eval_indices, test_indices


# ------------------------------
# Conformal Predictor & Core Evaluation Functions
# ------------------------------
class SplitPredictorFromProbabilities:
    """A wrapper for torchcp score functions to work directly with probabilities."""

    def __init__(self, score_function):
        self.score_function = score_function
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_hat = None

    def calibrate(self, cal_probabilities, cal_labels, alpha):
        """Calibrate the predictor to find q_hat."""
        cal_probabilities = cal_probabilities.to(self._device)
        cal_labels = cal_labels.to(self._device)
        with torch.no_grad():
            logits = torch.log(torch.clamp(cal_probabilities, 1e-8, 1.0))
            scores = self.score_function(logits, cal_labels)
            self.q_hat = calculate_conformal_value(scores, alpha)
        return self.q_hat.item()

    def predict(self, probabilities, q_hat_val):
        """Make predictions using a given q_hat value."""
        probabilities = probabilities.to(self._device)
        q_hat_tensor = torch.tensor(q_hat_val).to(self._device)
        with torch.no_grad():
            logits = torch.log(torch.clamp(probabilities, 1e-8, 1.0))
            scores = self.score_function(logits).to(self._device)
            return (scores <= q_hat_tensor).float()


def get_conformal_predictor(method_name):
    """Factory function to get a conformal predictor instance."""
    from torchcp.classification.score import APS, RAPS, TOPK, SAPS, THRRANK
    if method_name == "APS": return SplitPredictorFromProbabilities(score_function=APS())
    if method_name == "RAPS": return SplitPredictorFromProbabilities(score_function=RAPS())
    if method_name == "TOPK": return SplitPredictorFromProbabilities(score_function=TOPK())
    if method_name == "SAPS": return SplitPredictorFromProbabilities(score_function=SAPS())
    if method_name == "RANK": return SplitPredictorFromProbabilities(score_function=THRRANK())
    raise ValueError(f"Unknown CP method: {method_name}")


def evaluate_performance(predictor, q_hat, probabilities, labels, alpha, batch_size):
    """Evaluate performance (coverage, size, sscv) for a given dataset."""
    all_pred_sets, all_labels_list = [], []
    num_samples = len(probabilities)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, num_samples)
        batch_probs = probabilities[start:end]
        batch_labels = labels[start:end]

        pred_sets = predictor.predict(batch_probs, q_hat)
        all_pred_sets.extend(pred_sets.cpu().numpy())
        all_labels_list.extend(batch_labels.cpu().numpy())

    size_array = np.array([ps.sum() for ps in all_pred_sets])
    correct_array = np.array([ps[int(label)] for ps, label in zip(all_pred_sets, all_labels_list)])

    coverage = np.mean(correct_array)
    size = np.mean(size_array)

    stratified_size = [[0, 1], [2, 3], [4, 10], [11, 100], [101, 1000]]
    stratum_violations = []
    for stratum in stratified_size:
        idx = np.where((size_array >= stratum[0]) & (size_array <= stratum[1]))[0]
        if len(idx) > 0:
            stratum_violations.append(abs((1 - alpha) - np.mean(correct_array[idx])))
    sscv = max(stratum_violations) if stratum_violations else -1

    return coverage, size, sscv


# ------------------------------
# Nash Equilibrium & Simulation Logic
# ------------------------------
def find_nash_equilibrium(payoff_matrix):
    """Find Nash Equilibrium, handling potential NaNs."""
    if np.isnan(payoff_matrix).any():
        max_val = np.nanmax(payoff_matrix)
        payoff_matrix_safe = np.nan_to_num(payoff_matrix, nan=max_val if not np.isnan(max_val) else 1e6)
    else:
        payoff_matrix_safe = payoff_matrix

    game = nash.Game(-payoff_matrix_safe, payoff_matrix_safe)
    try:
        equilibria = list(game.support_enumeration())
        return equilibria
    except Exception as e:
        print(f"      - Nash equilibrium computation failed: {e}")
        return []


def select_best_nash_equilibrium(equilibria):
    """Select the equilibrium with the largest combined support (most mixed)."""
    if not equilibria: return None
    return max(equilibria, key=lambda eq: np.sum(eq[0] > 1e-5) + np.sum(eq[1] > 1e-5))


def evaluate_all_strategy_pairs(test_indices, model_names, attacks, cp_method, qhats, alpha,
                                nash_defender_strategy, nash_attacker_strategy,
                                adv_predictions_path, dataset_name, batch_size):
    """
    Calculates performance for a full matrix of defense vs. attack strategies.
    - 7 Defense Strategies: 5 single models, 1 Nash, 1 Uniform.
    - 10 Attack Strategies: 9 single attacks, 1 Nash.
    """
    print("    Phase 4: Starting comprehensive strategy evaluation on the test set...")
    predictor = get_conformal_predictor(cp_method)

    print("      - Pre-calculating base performance metrics (Coverage, Size, SSCV)...")
    base_performance = {}
    for model in tqdm(model_names, desc="        Models", leave=False):
        base_performance[model] = {}
        for attack in tqdm(attacks, desc="        Attacks", leave=False):
            probs, labels = load_model_predictions(adv_predictions_path, dataset_name, model, attack)
            if probs is not None:
                q_hat_val = qhats[model]
                coverage, size, sscv = evaluate_performance(predictor, q_hat_val, probs[test_indices],
                                                            labels[test_indices], alpha, batch_size)
                base_performance[model][attack] = {'coverage': coverage, 'size': size, 'sscv': sscv}
            else:
                base_performance[model][attack] = {'coverage': np.nan, 'size': np.nan, 'sscv': np.nan}

    defense_strategies = {name: np.zeros(len(model_names)) for name in model_names}
    for i, name in enumerate(model_names): defense_strategies[name][i] = 1.0
    defense_strategies['Nash'] = nash_defender_strategy
    defense_strategies['Uniform'] = np.ones(len(model_names)) / len(model_names)

    attack_strategies = {name: np.zeros(len(attacks)) for name in attacks}
    for i, name in enumerate(attacks): attack_strategies[name][i] = 1.0
    attack_strategies['Nash'] = nash_attacker_strategy

    print("      - Calculating expected outcomes for all 7x10 strategy pairings...")
    final_results = []
    for def_name, def_probs in tqdm(defense_strategies.items(), desc="      Defense Strategies", leave=False):
        for atk_name, atk_probs in tqdm(attack_strategies.items(), desc="      Attack Strategies", leave=False):
            expected_coverage, expected_size, expected_sscv = 0.0, 0.0, 0.0
            total_prob = 0.0
            for i, model in enumerate(model_names):
                for j, attack in enumerate(attacks):
                    prob = def_probs[i] * atk_probs[j]
                    if prob > 0 and model in base_performance and attack in base_performance[model]:
                        perf = base_performance[model][attack]
                        if not np.isnan(perf['size']):
                            expected_size += prob * perf['size']
                            expected_coverage += prob * perf['coverage']
                            expected_sscv += prob * perf['sscv']
                            total_prob += prob
            if total_prob > 0:
                expected_size /= total_prob
                expected_coverage /= total_prob
                expected_sscv /= total_prob
            else:
                expected_size, expected_coverage, expected_sscv = np.nan, np.nan, np.nan

            final_results.append({
                'defense_strategy': def_name,
                'attack_strategy': atk_name,
                'expected_size': expected_size,
                'expected_coverage': expected_coverage,
                'expected_sscv': expected_sscv
            })
    gc.collect()
    return final_results


# ------------------------------
# Main Experiment Logic
# ------------------------------
def run_single_experiment(dataset_name, model_names, cp_methods, attacks,
                          adv_predictions_path, alpha, experiment_id, batch_size, save_dir):
    exp_seed = 42 + experiment_id
    set_seed(exp_seed)

    ref_probs, _ = load_model_predictions(adv_predictions_path, dataset_name, model_names[0], 'clean')
    if ref_probs is None:
        print(f"Could not load reference predictions for {dataset_name}. Skipping experiment.")
        return

    cal_indices, eval_indices, test_indices = split_dataset_three_way(len(ref_probs), seed=exp_seed)
    print(f"\n--- Exp {experiment_id + 1}/{args.num_experiments} (Seed: {exp_seed}) on {dataset_name.upper()} ---")
    print(f"    Data split: Cal={len(cal_indices)}, Eval={len(eval_indices)}, Test={len(test_indices)}")

    for cp_method in cp_methods:
        print(f"\n  Processing CP Method: {cp_method.upper()}")

        results_dir = os.path.join(save_dir, dataset_name, cp_method)
        os.makedirs(results_dir, exist_ok=True)

        predictor = get_conformal_predictor(cp_method)

        print("    Phase 2: Calibrating to find robust q_hat for each model...")
        qhats = {}
        for model in model_names:
            qhat_list = []
            for attack in attacks:
                probs, labels = load_model_predictions(adv_predictions_path, dataset_name, model, attack)
                if probs is not None:
                    qhat_list.append(predictor.calibrate(probs[cal_indices], labels[cal_indices], alpha))
            if qhat_list:
                qhats[model] = max(qhat_list)
        print(f"      - Calibrated q_hats: {{ {', '.join([f'{k}: {v:.4f}' for k, v in qhats.items()])} }}")

        print("    Phase 3: Building payoff matrix on EVALUATION set...")
        eval_size_matrix = pd.DataFrame(index=model_names, columns=attacks, dtype=float)
        for model, q_hat_val in qhats.items():
            for attack in attacks:
                probs, labels = load_model_predictions(adv_predictions_path, dataset_name, model, attack)
                if probs is not None:
                    _, size, _ = evaluate_performance(predictor, q_hat_val, probs[eval_indices], labels[eval_indices],
                                                      alpha, batch_size)
                    eval_size_matrix.loc[model, attack] = size

        # --- NEW CODE BLOCK: Build payoff matrix on TEST set ---
        print("    Phase 3.5: Building payoff matrix on TEST set...")
        test_size_matrix = pd.DataFrame(index=model_names, columns=attacks, dtype=float)
        for model, q_hat_val in qhats.items():
            for attack in attacks:
                probs, labels = load_model_predictions(adv_predictions_path, dataset_name, model, attack)
                if probs is not None:
                    # The ONLY change is using test_indices here
                    _, size, _ = evaluate_performance(predictor, q_hat_val, probs[test_indices], labels[test_indices],
                                                      alpha, batch_size)
                    test_size_matrix.loc[model, attack] = size
        # --- END NEW CODE BLOCK ---

        print("      - Finding Nash Equilibrium...")
        equilibria = find_nash_equilibrium(eval_size_matrix.to_numpy(dtype=float))
        if not equilibria:
            print(f"      - No Nash equilibrium found for {cp_method}. Skipping.")
            continue

        nash_eq = select_best_nash_equilibrium(equilibria)
        defender_strategy, attacker_strategy = nash_eq
        print(f"      - Nash Defender Strategy: {np.round(defender_strategy, 3)}")
        print(f"      - Nash Attacker Strategy: {np.round(attacker_strategy, 3)}")

        # --- MODIFIED: Save ALL required data to the Excel file ---
        nash_df_path = os.path.join(results_dir, f"exp{experiment_id + 1}_nash_data.xlsx")
        defender_df = pd.DataFrame(defender_strategy, index=model_names, columns=['Probability'])
        attacker_df = pd.DataFrame(attacker_strategy, index=attacks, columns=['Probability'])
        with pd.ExcelWriter(nash_df_path) as writer:
            eval_size_matrix.to_excel(writer, sheet_name='Payoff_Matrix_Size')
            test_size_matrix.to_excel(writer, sheet_name='Payoff_Matrix_Size_Test')  # Save the new test matrix
            defender_df.to_excel(writer, sheet_name='Defender_Nash_Strategy')
            attacker_df.to_excel(writer, sheet_name='Attacker_Nash_Strategy')
        print(f"      - Payoff matrices (Eval & Test) and Nash strategies saved to: {nash_df_path}")
        # --- END MODIFICATION ---

        # Phase 4 (evaluate_all_strategy_pairs) is now optional for plotting, but we can keep it for CSV data.
        final_results_list = evaluate_all_strategy_pairs(
            test_indices, model_names, attacks, cp_method, qhats, alpha,
            defender_strategy, attacker_strategy, adv_predictions_path, dataset_name, batch_size
        )

        if final_results_list:
            results_df = pd.DataFrame(final_results_list)
            results_df['experiment_id'] = experiment_id + 1
            results_df['cp_method'] = cp_method
            results_df['dataset'] = dataset_name

            eval_results_path = os.path.join(results_dir, f"exp{experiment_id + 1}_evaluation.csv")
            results_df.to_csv(eval_results_path, index=False)
            print(f"    Comprehensive evaluation results saved to: {eval_results_path}")


# ------------------------------
# Main Program Driver
# ------------------------------
def main(args):
    FIXED_MODELS = ['Chen2024Data_WRN_50_2', 'Debenedetti2022Light_XCiT-M12', 'Peng2023Robust']
    FIXED_ATTACKS = ['APGD', 'CW', 'FGSM', 'GN', 'PGD', 'PIFGSM', 'clean']

    os.makedirs(args.save_dir, exist_ok=True)

    for dataset_name in args.datasets:
        print(f"\n{'=' * 80}\nProcessing Dataset: {dataset_name.upper()}\n{'=' * 80}")

        for exp_id in range(args.num_experiments):
            try:
                run_single_experiment(
                    dataset_name, FIXED_MODELS, args.cp_methods, FIXED_ATTACKS,
                    args.adv_predictions_path, args.alpha, exp_id, args.batch_size, args.save_dir
                )
            except Exception as e:
                logging.error(f"Dataset {dataset_name}, Exp {exp_id + 1} failed: {e}", exc_info=True)
                print(f"FATAL ERROR in Exp {exp_id + 1} for {dataset_name}: {e}")

    print(f"\n{'=' * 80}\nAll datasets processing completed!\nResults saved in: {args.save_dir}\n{'=' * 80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rigorous Multi-CP Game Theory Analysis")
    parser.add_argument('--datasets', nargs='+', default=['imagenet'], help='Datasets')
    parser.add_argument('--num_experiments', default=20, type=int, help='Number of experiments')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--alpha', default=0.1, type=float, help='Error rate')
    parser.add_argument('--adv_predictions_path', default="./adv_predictions", type=str, help='Path to predictions')
    parser.add_argument('--cp_methods', nargs='+', default=['APS', 'RAPS', 'TOPK', 'SAPS', 'RANK'], help='CP methods')
    parser.add_argument('--save_dir', default="./game_theory_results", type=str,
                        help='Main save directory for all results')
    args = parser.parse_args()
    main(args)