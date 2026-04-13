# AML ML Models Implementation

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score, average_precision_score, roc_curve, auc
)
import xgboost as xgb
from jellyfish import jaro_winkler_similarity, levenshtein_distance, soundex
import matplotlib.pyplot as plt
import random
import string
from joblib import dump, load
import warnings
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
warnings.filterwarnings("ignore")


class AMLMLModels:
    def __init__(self, training_data_path=None, training_df=None):
        if training_data_path:
            self.df = pd.read_excel(training_data_path)
        elif training_df is not None:
            self.df = training_df
        else:
            raise ValueError("Either training_data_path or training_df must be provided")
        self.models = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = [
            "jaro_winkler_score", "levenshtein_score", "soundex_match",
            "is_high_risk", "is_medium_risk", "has_nationality_match",
        ]
        self.decision_mapping = {"ALLOWED": 0, "AMBIGUOUS": 1, "BLOCKED": 2}
        print("AML ML Models initialized successfully!")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Decision distribution:\n{self.df['decision'].value_counts()}")

    def prepare_data(self, test_size=0.2, random_state=42):
        print("\nPreparing data...")
        self.df = self.df.dropna(subset=self.feature_columns + ["decision"])
        self.df = self.df[self.df["decision"].isin(self.decision_mapping.keys())]
        self.df["decision"] = self.df["decision"].map(self.decision_mapping)
        X = self.df[self.feature_columns]
        y = self.df["decision"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=self.feature_columns)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.feature_columns)
        print(f"Training set shape: {self.X_train.shape}, Test set shape: {self.X_test.shape}")
        print(f"Training decision distribution:\n{self.y_train.value_counts()}")
        print(f"Test decision distribution:\n{self.y_test.value_counts()}")

    def add_label_noise(self, noise_rate=0.05):
        n_samples = len(self.y_train)
        n_flip = int(n_samples * noise_rate)
        flip_indices = np.random.choice(n_samples, n_flip, replace=False)
        classes = [0, 1, 2]
        for idx in flip_indices:
            current = self.y_train.iloc[idx]
            new = random.choice([c for c in classes if c != current])
            self.y_train.iloc[idx] = new
        print(f"Added label noise to {n_flip} training samples.")

    def train_random_forest(self, random_state=42):
        print("\n=== TRAINING RANDOM FOREST ===")
        class_weights = "balanced"
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5,
            random_state=random_state, class_weight=class_weights
        )
        rf_model.fit(self.X_train, self.y_train)
        self.models["Random Forest"] = rf_model
        feature_importance = pd.DataFrame(
            {"feature": self.feature_columns, "importance": rf_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        print("Feature Importance (Random Forest):")
        print(feature_importance)
        return rf_model

    def train_logistic_regression(self, random_state=42):
        print("\n=== TRAINING LOGISTIC REGRESSION ===")
        lr_model = LogisticRegression(
            random_state=random_state, class_weight="balanced", max_iter=1000, C=0.1
        )
        lr_model.fit(self.X_train, self.y_train)
        self.models["Logistic Regression"] = lr_model
        if hasattr(lr_model, "coef_"):
            coef_df = pd.DataFrame(
                lr_model.coef_.T, columns=[f"Class_{i}" for i in range(lr_model.coef_.shape[0])],
                index=self.feature_columns
            )
            print("Feature Coefficients (Logistic Regression):")
            print(coef_df)
        return lr_model

    def train_xgboost(self, random_state=42):
        print("\n=== TRAINING XGBOOST ===")
        class_counts = np.bincount(self.y_train)
        class_weights = len(self.y_train) / (len(class_counts) * class_counts)
        sample_weights = class_weights[self.y_train]
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=random_state, stratify=self.y_train
        )
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000, max_depth=3, learning_rate=0.05,
            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
            random_state=random_state, eval_metric="mlogloss"
        )
        try:
            # Use callback for early stopping
            early_stop = xgb.callback.EarlyStopping(rounds=50, metric_name="mlogloss", save_best=True)
            xgb_model.fit(
                X_tr, y_tr, sample_weight=class_weights[y_tr],
                eval_set=[(X_val, y_val)], callbacks=[early_stop], verbose=False
            )
            print("XGBoost trained with early stopping callback")
        except:
            # Fallback for older versions or if callback fails
            xgb_model.fit(X_tr, y_tr, sample_weight=class_weights[y_tr], verbose=False)
            print("XGBoost trained without early stopping (fallback)")
        self.models["XGBoost"] = xgb_model
        feature_importance = pd.DataFrame(
            {"feature": self.feature_columns, "importance": xgb_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        print("Feature Importance (XGBoost):")
        print(feature_importance)

        feature_importance.plot(kind='barh', x='feature', y='importance', legend=False, color='#4C72B0')
        plt.title('XGBoost Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('fig_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.clf()

        return xgb_model

    def evaluate_model(self, model_name, model, use_scaled_data=True):
        print(f"\n=== EVALUATING {model_name.upper()} ===")
        X_test_data = self.X_test
        y_pred = model.predict(X_test_data)
        y_pred_proba = model.predict_proba(X_test_data)
        accuracy = accuracy_score(self.y_test, y_pred)
        macro_f1 = classification_report(self.y_test, y_pred, output_dict=True)['macro avg']['f1-score']
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=["ALLOWED", "AMBIGUOUS", "BLOCKED"]))
        cm = confusion_matrix(self.y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        auc_score = roc_auc_score(self.y_test, y_pred_proba, multi_class="ovr")
        pr_auc = average_precision_score(label_binarize(self.y_test, classes=[0,1,2]), y_pred_proba, average="macro")
        print(f"AUC Score (ovr): {auc_score:.4f}")
        print(f"PR-AUC (macro): {pr_auc:.4f}")
        y_train_pred = model.predict(self.X_train)
        train_acc = accuracy_score(self.y_train, y_train_pred)
        print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {accuracy:.4f}, Gap: {train_acc - accuracy:.4f}")
        return {"accuracy": accuracy, "macro_f1": macro_f1, "auc": auc_score, "pr_auc": pr_auc}

    def train_all_models(self):
        print("\n" + "=" * 50)
        print("TRAINING ALL ML MODELS")
        print("=" * 50)
        self.add_label_noise()
        self.train_random_forest()
        self.train_logistic_regression()
        self.train_xgboost()
        print(f"\nAll models trained successfully! Models available: {list(self.models.keys())}")

    def evaluate_all_models(self):
        print("\n" + "=" * 50)
        print("EVALUATING ALL MODELS")
        print("=" * 50)
        results = {}
        for model_name, model in self.models.items():
            results[model_name] = self.evaluate_model(model_name, model)

        import seaborn as sns
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]
        class_names = ["ALLOWED", "AMBIGUOUS", "BLOCKED"]
        for ax, (model_name, model) in zip(axes, self.models.items()):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                        xticklabels=class_names, yticklabels=class_names,
                        cmap='Blues')
            ax.set_title(model_name)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        plt.tight_layout()
        plt.savefig('fig_confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.clf()

        print("\n=== CROSS-VALIDATION RESULTS (on Train) ===")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for model_name, model in self.models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring='accuracy')
            print(f"{model_name} CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return results

    def predict_new_case(self, features_dict):
        print("\n=== PREDICTING NEW CASE ===")
        feature_vector = np.array([features_dict.get(col, 0) for col in self.feature_columns]).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        predictions = {}
        reverse_mapping = {v: k for k, v in self.decision_mapping.items()}
        for model_name, model in self.models.items():
            pred = model.predict(feature_vector_scaled)[0]
            pred_proba = model.predict_proba(feature_vector_scaled)[0]
            predictions[model_name] = {
                "decision": reverse_mapping[pred],
                "probabilities": {"ALLOWED": pred_proba[0], "AMBIGUOUS": pred_proba[1], "BLOCKED": pred_proba[2]},
            }
        print("Input features:")
        for col, val in zip(self.feature_columns, feature_vector[0]):
            print(f" {col}: {val}")
        print("\nPredictions:")
        for model_name, result in predictions.items():
            print(f"\n{model_name}:")
            print(f" Decision: {result['decision']}")
            print(f" Probabilities:")
            for decision, prob in result["probabilities"].items():
                print(f" {decision}: {prob:.4f}")
        return predictions

    def save_models(self, filepath_prefix="aml_model"):
        for model_name, model in self.models.items():
            filename = f"{filepath_prefix}_{model_name.lower().replace(' ', '_')}.joblib"
            dump(model, filename)
            print(f"Saved {model_name} to {filename}")
        scaler_filename = f"{filepath_prefix}_scaler.joblib"
        dump(self.scaler, scaler_filename)
        print(f"Saved scaler to {scaler_filename}")

    def load_models(self, filepath_prefix="aml_model"):
        model_files = {
            "Random Forest": f"{filepath_prefix}_random_forest.joblib",
            "Logistic Regression": f"{filepath_prefix}_logistic_regression.joblib",
            "XGBoost": f"{filepath_prefix}_xgboost.joblib",
        }
        for model_name, filename in model_files.items():
            try:
                self.models[model_name] = load(filename)
                print(f"Loaded {model_name} from {filename}")
            except:
                print(f"Could not load {model_name} from {filename}")
        scaler_filename = f"{filepath_prefix}_scaler.joblib"
        try:
            self.scaler = load(scaler_filename)
            print(f"Loaded scaler from {scaler_filename}")
        except:
            print(f"Could not load scaler from {scaler_filename}")

    def plot_all_models_roc(self):
        if not self.models:
            print("No models to plot.")
            return
        y_test_bin = label_binarize(self.y_test, classes=[0, 1, 2])
        n_classes = y_test_bin.shape[1]
        class_names = ["ALLOWED", "AMBIGUOUS", "BLOCKED"]
        colors = ["blue", "orange", "green"]
        linestyles = ["-", "--", ":"]
        plt.figure(figsize=(10, 8))
        for model_idx, (model_name, model) in enumerate(self.models.items()):
            y_score = model.predict_proba(self.X_test)
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                auc = roc_auc_score(y_test_bin[:, i], y_score[:, i])
                plt.plot(
                    fpr, tpr, label=f"{model_name} - {class_names[i]} (AUC={auc:.2f})",
                    color=colors[i], linestyle=linestyles[model_idx % len(linestyles)],
                )
        plt.plot([0, 1], [0, 1], "k--", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for All Models and Classes")
        plt.legend(loc="lower right", fontsize="small")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('fig_roc_curves.png', dpi=150, bbox_inches='tight')
        plt.show()

class AMLTrainingDataGenerator:
    def __init__(self, watchlist_df):
        self.watchlist_df = watchlist_df
        self.training_data = []

    def generate_name_variations(self, original_name, num_variations=5):
        variations = []
        variations.append(original_name)
        for _ in range(2):
            name_chars = list(original_name)
            if len(name_chars) > 3:
                pos = random.randint(0, len(name_chars) - 1)
                if name_chars[pos] != " ":
                    name_chars[pos] = random.choice(string.ascii_lowercase)
            variations.append("".join(name_chars))
        for _ in range(2):
            name_chars = list(original_name)
            if len(name_chars) > 3:
                pos = random.randint(0, len(name_chars) - 1)
                if name_chars[pos] != " ":
                    name_chars.pop(pos)
            variations.append("".join(name_chars))
        words = original_name.split()
        if len(words) > 1:
            variations.append(" ".join(reversed(words)))
            if len(words) > 2:
                shuffled = words.copy()
                random.shuffle(shuffled)
                variations.append(" ".join(shuffled))
        if len(words) > 1:
            variations.append(words[0])
            variations.append(words[-1])
        return variations[:num_variations]

    def generate_random_names(self, num_names=100):
        first_names = [
            "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa", "James", "Mary",
            "William", "Patricia", "Richard", "Jennifer", "Charles", "Ahmed", "Fatima", "Omar", "Aisha",
            "Hassan", "Zainab", "Ali", "Nour",
        ]
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez",
            "Martinez", "Hernandez", "Lopez", "Gonzalez", "Al-Rashid", "Ben-Ali", "Mahmoud", "Hassan",
            "Ibrahim", "Khalil",
        ]
        random_names = []
        for _ in range(num_names):
            first = random.choice(first_names)
            last = random.choice(last_names)
            random_names.append(f"{first} {last}")
        return random_names

    def calculate_features(self, input_name, watchlist_row):
        dataset_name = watchlist_row["Full Name"]
        name1 = input_name.lower().strip()
        name2 = dataset_name.lower().strip()
        jaro_winkler_score = jaro_winkler_similarity(name1, name2) * 100
        max_len = max(len(name1), len(name2))
        levenshtein_score = (1 - levenshtein_distance(name1, name2) / max_len) * 100 if max_len > 0 else 100
        soundex_match = 1 if soundex(name1) == soundex(name2) else 0
        risk_category = watchlist_row["Risk Category"]
        is_high_risk = 1 if risk_category in ["Terrorism Financing", "Money Laundering"] else 0
        is_medium_risk = 1 if risk_category == "PEP" else 0
        has_nationality_match = random.choice([0, 1])
        return {
            "jaro_winkler_score": round(jaro_winkler_score, 2),
            "levenshtein_score": round(levenshtein_score, 2),
            "soundex_match": soundex_match,
            "is_high_risk": is_high_risk,
            "is_medium_risk": is_medium_risk,
            "has_nationality_match": has_nationality_match,
            "risk_category": risk_category,
            "best_match_name": dataset_name,
        }

    def find_best_match(self, input_name):
        best_match = None
        best_score = 0
        for _, row in self.watchlist_df.iterrows():
            features = self.calculate_features(input_name, row)
            combined_score = (
                features["jaro_winkler_score"] * 0.4 +
                features["levenshtein_score"] * 0.3 +
                features["soundex_match"] * 100 * 0.1 +
                features["has_nationality_match"] * 100 * 0.2
            )
            if combined_score > best_score:
                best_score = combined_score
                best_match = features
                best_match["combined_score"] = round(combined_score, 2)
        return best_match

    def apply_decision_rules(self, features):
        score = features["combined_score"]
        if features["is_high_risk"] == 1:
            risk_level = "High Risk"
        elif features["is_medium_risk"] == 1:
            risk_level = "Medium Risk"
        else:
            risk_level = "Unknown Risk"
        if score >= 80 and risk_level in ["High Risk", "Medium Risk"]:
            decision = "BLOCKED"
            reason = "High similarity match with high/medium risk customer"
        elif 40 <= score < 80 and risk_level == "High Risk":
            decision = "BLOCKED"
            reason = "High-risk match found"
        elif 40 <= score < 80 and risk_level == "Medium Risk":
            decision = "AMBIGUOUS"
            reason = "Medium-risk match found - manual review required"
        elif score < 20:
            decision = "ALLOWED"
            reason = "No major matches found"
        else:
            decision = "ALLOWED"
            reason = "No significant risk identified"
        return decision, reason

    def generate_training_dataset(self, num_variations_per_watchlist=3, num_random_names=200):
        print("Generating training dataset...")
        print("Processing watchlist variations...")
        for _, row in self.watchlist_df.iterrows():
            original_name = row["Full Name"]
            variations = self.generate_name_variations(original_name, num_variations_per_watchlist)
            for variation in variations:
                best_match = self.find_best_match(variation)
                decision, reason = self.apply_decision_rules(best_match)
                correct_decision = random.choices([0, 1], weights=[0.2, 0.8])[0]
                training_record = {
                    "input_name": variation,
                    "jaro_winkler_score": best_match["jaro_winkler_score"],
                    "levenshtein_score": best_match["levenshtein_score"],
                    "soundex_match": best_match["soundex_match"],
                    "is_high_risk": best_match["is_high_risk"],
                    "is_medium_risk": best_match["is_medium_risk"],
                    "has_nationality_match": best_match["has_nationality_match"],
                    "combined_score": best_match["combined_score"],
                    "best_match_name": best_match["best_match_name"],
                    "risk_category": best_match["risk_category"],
                    "decision": decision,
                    "reason": reason,
                    "correct_decision": correct_decision,
                }
                self.training_data.append(training_record)
        print("Processing random names...")
        random_names = self.generate_random_names(num_random_names)
        for name in random_names:
            best_match = self.find_best_match(name)
            decision, reason = self.apply_decision_rules(best_match)
            # Force lower similarity scores for ALLOWED to reduce correlation
            if decision == "ALLOWED":
                best_match["jaro_winkler_score"] = min(best_match["jaro_winkler_score"], 20.0)
                best_match["levenshtein_score"] = min(best_match["levenshtein_score"], 20.0)
                best_match["combined_score"] = min(best_match["combined_score"], 20.0)
            correct_decision = 1 if decision == "ALLOWED" else 0
            training_record = {
                "input_name": name,
                "jaro_winkler_score": best_match["jaro_winkler_score"],
                "levenshtein_score": best_match["levenshtein_score"],
                "soundex_match": best_match["soundex_match"],
                "is_high_risk": best_match["is_high_risk"],
                "is_medium_risk": best_match["is_medium_risk"],
                "has_nationality_match": best_match["has_nationality_match"],
                "combined_score": best_match["combined_score"],
                "best_match_name": best_match["best_match_name"],
                "risk_category": best_match["risk_category"],
                "decision": decision,
                "reason": reason,
                "correct_decision": correct_decision,
            }
            self.training_data.append(training_record)
        df = pd.DataFrame(self.training_data)
        n_flip = int(len(df) * 0.05)
        flip_indices = np.random.choice(len(df), n_flip, replace=False)
        classes = ["ALLOWED", "AMBIGUOUS", "BLOCKED"]
        for idx in flip_indices:
            current = df.at[idx, 'decision']
            new = random.choice([c for c in classes if c != current])
            df.at[idx, 'decision'] = new
        print(f"Added label noise to {n_flip} samples in generated dataset.")
        return df

def main():
    print("Step 1: Load or generate training dataset...")
    training_data_path = r"C:\Users\ahmed\OneDrive\Desktop\aml-name-screening_project-main\aml-name-screening_project-main\dataset\cleaned_aml_data.xlsx"
    try:
        watchlist_df = pd.read_excel(training_data_path)
        watchlist_df = watchlist_df.dropna(subset=['Full Name', 'Risk Category'])
        print(f"Loaded watchlist: {watchlist_df.shape}")
    except Exception as e:
        print(f"Could not load {training_data_path}: {e}. Generating new dataset...")
        watchlist_df = pd.DataFrame({
            "Full Name": ["John Doe", "Ahmed Hassan"],
            "Risk Category": ["Terrorism Financing", "PEP"],
            "Nationality": ["USA", "Syria"]
        })
            
    import matplotlib.pyplot as plt
    risk_counts = watchlist_df['Risk Category'].value_counts()
    plt.figure()
    risk_counts.plot(kind='bar', color=['#4C72B0','#DD8452','#55A868'])
    plt.title('Risk Category Distribution')
    plt.xlabel('Risk Category')
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('fig_risk_dist.png', dpi=150, bbox_inches='tight')
    plt.clf()
        
    watchlist_sample = watchlist_df.sample(n=300, random_state=42).reset_index(drop=True)
    generator = AMLTrainingDataGenerator(watchlist_sample)
    training_df = generator.generate_training_dataset(
        num_variations_per_watchlist=3,
        num_random_names=200
    )

    decision_counts = training_df['decision'].value_counts()
    plt.figure()
    decision_counts.plot(kind='bar', color=['#55A868','#DD8452','#C44E52'])
    plt.title('Generated Training Decision Labels')
    plt.xlabel('Decision')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('fig_decision_dist.png', dpi=150, bbox_inches='tight')
    plt.clf()

    print(f"Decision distribution:\n{training_df['decision'].value_counts()}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for decision, group in training_df.groupby('decision'):
        axes[0].hist(group['jaro_winkler_score'], bins=20, alpha=0.6, label=decision)
        axes[1].hist(group['levenshtein_score'],  bins=20, alpha=0.6, label=decision)
    axes[0].set_title('Jaro-Winkler Score by Decision')
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[1].set_title('Levenshtein Score by Decision')
    axes[1].set_xlabel('Score')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('fig_feature_hist.png', dpi=150, bbox_inches='tight')
    plt.clf()

    # Balance classes
    allowed_df = training_df[training_df["decision"] == "ALLOWED"]
    ambiguous_df = training_df[training_df["decision"] == "AMBIGUOUS"]
    blocked_df = training_df[training_df["decision"] == "BLOCKED"]
    if len(allowed_df) == 0:
        print("Warning: No ALLOWED samples. Generating dummy ALLOWED samples...")
        for i in range(100):
            dummy_row = {
                "input_name": f"dummy_allowed_{i}",
                "jaro_winkler_score": 10.0,
                "levenshtein_score": 10.0,
                "soundex_match": 0,
                "is_high_risk": 0,
                "is_medium_risk": 0,
                "has_nationality_match": 0,
                "combined_score": 10.0,
                "best_match_name": "None",
                "risk_category": "None",
                "decision": "ALLOWED",
                "reason": "Dummy allowed sample",
                "correct_decision": 1,
            }
            training_df = pd.concat([training_df, pd.DataFrame([dummy_row])], ignore_index=True)
        allowed_df = training_df[training_df["decision"] == "ALLOWED"]
    max_size = max(len(allowed_df), len(ambiguous_df), len(blocked_df))
    allowed_upsampled = resample(allowed_df, replace=True, n_samples=max_size, random_state=42)
    ambiguous_upsampled = resample(ambiguous_df, replace=True, n_samples=max_size, random_state=42)
    blocked_upsampled = resample(blocked_df, replace=True, n_samples=max_size, random_state=42)
    training_df = pd.concat([allowed_upsampled, ambiguous_upsampled, blocked_upsampled])
    training_df = training_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Balanced class distribution:\n{training_df['decision'].value_counts()}")
    print("\nStep 2: Initialize ML Models...")
    ml_models = AMLMLModels(training_df=training_df)
    ml_models.prepare_data()
    ml_models.train_all_models()
    results = ml_models.evaluate_all_models()
    print("\nStep 3: Test prediction on new case...")
    new_case = {
        "jaro_winkler_score": 85.5,
        "levenshtein_score": 82.3,
        "soundex_match": 1,
        "is_high_risk": 1,
        "is_medium_risk": 0,
        "has_nationality_match": 0,
    }
    predictions = ml_models.predict_new_case(new_case)
    print("\nStep 4: Save models...")
    ml_models.save_models()
    print("\n" + "=" * 50)
    print("ML MODEL TRAINING COMPLETE!")
    print("=" * 50)
    return ml_models, results

if __name__ == "__main__":
    ml_models, results = main()
    if ml_models is not None:
        ml_models.plot_all_models_roc()