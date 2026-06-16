"""Native multinomial tier-conversion example for insurance pricing."""

from __future__ import annotations

import numpy as np
import polars as pl
import rustystats as rs

CLASSES = ["none", "basic", "standard", "premium"]


def make_quotes(n: int = 1_200, seed: int = 7) -> pl.DataFrame:
    """Build a synthetic quote table with tier prices and availability."""

    rng = np.random.default_rng(seed)
    age = rng.normal(42.0, 12.0, size=n).clip(18.0, 80.0)
    vehicle_value = rng.lognormal(mean=10.0, sigma=0.35, size=n)
    channel = rng.choice(["direct", "agent", "partner"], p=[0.45, 0.35, 0.20], size=n)
    channel_agent = (channel == "agent").astype(float)
    channel_partner = (channel == "partner").astype(float)

    scaled_age = (age - 42.0) / 12.0
    scaled_value = (np.log(vehicle_value) - 10.0) / 0.35

    price_basic = np.exp(np.log(320.0) + 0.08 * scaled_age + rng.normal(scale=0.20, size=n))
    price_standard = np.exp(np.log(460.0) + 0.06 * scaled_age + rng.normal(scale=0.20, size=n))
    price_premium = np.exp(np.log(640.0) + 0.04 * scaled_age + rng.normal(scale=0.20, size=n))

    richness_basic = 1.00 + 0.05 * scaled_value + rng.normal(scale=0.08, size=n)
    richness_standard = 1.35 + 0.08 * scaled_value + rng.normal(scale=0.08, size=n)
    richness_premium = 1.75 + 0.12 * scaled_value + rng.normal(scale=0.08, size=n)
    premium_available = (vehicle_value > 16_000.0) | (channel != "partner")
    quote_weight = np.where(channel == "agent", 1.25, 1.0)

    eta = np.column_stack(
        [
            np.zeros(n),
            6.0
            + 0.30 * scaled_age
            - 0.25 * scaled_value
            - 0.10 * channel_agent
            - 1.05 * np.log(price_basic)
            + 0.25 * richness_basic,
            6.4
            + 0.10 * scaled_age
            + 0.20 * scaled_value
            + 0.35 * channel_agent
            - 1.05 * np.log(price_standard)
            + 0.45 * richness_standard,
            7.0
            - 0.15 * scaled_age
            + 0.45 * scaled_value
            + 0.25 * channel_partner
            - 1.05 * np.log(price_premium)
            + 0.70 * richness_premium,
        ]
    )
    eta[~premium_available, CLASSES.index("premium")] = -np.inf
    exp_eta = np.exp(eta - np.max(eta, axis=1, keepdims=True))
    probabilities = exp_eta / exp_eta.sum(axis=1, keepdims=True)
    purchased_tier = [rng.choice(CLASSES, p=row) for row in probabilities]

    return pl.DataFrame(
        {
            "PurchasedTier": purchased_tier,
            "DriverAge": age,
            "VehicleValueLog": np.log(vehicle_value),
            "Channel": channel,
            "price_basic": price_basic,
            "price_standard": price_standard,
            "price_premium": price_premium,
            "richness_basic": richness_basic,
            "richness_standard": richness_standard,
            "richness_premium": richness_premium,
            "premium_available": premium_available,
            "quote_weight": quote_weight,
        }
    )


def weighted_log_loss(data: pl.DataFrame, probabilities: np.ndarray) -> float:
    class_to_idx = {label: idx for idx, label in enumerate(CLASSES)}
    y_codes = np.asarray([class_to_idx[label] for label in data["PurchasedTier"].to_list()])
    weights = data["quote_weight"].to_numpy()
    observed = np.clip(probabilities[np.arange(data.height), y_codes], np.finfo(float).tiny, 1.0)
    return float(-np.sum(weights * np.log(observed)) / np.sum(weights))


def format_mix(mix: dict[str, float]) -> str:
    return ", ".join(f"{label}={mix[label]:.1%}" for label in CLASSES)


def main() -> None:
    quotes = make_quotes()
    train = quotes.head(900)
    holdout = quotes.tail(300)

    result = rs.multinomial_dict(
        response="PurchasedTier",
        shared_terms={
            "DriverAge": {"type": "bs", "df": 5},
            "VehicleValueLog": {"type": "linear"},
            "Channel": {"type": "categorical"},
        },
        alternative_terms={
            "log_price": {
                "columns": {
                    "basic": "price_basic",
                    "standard": "price_standard",
                    "premium": "price_premium",
                },
                "coefficient": "generic",
                "transform": "log",
            },
            "richness": {
                "columns": {
                    "basic": "richness_basic",
                    "standard": "richness_standard",
                    "premium": "richness_premium",
                },
                "coefficient": "class_specific",
            },
        },
        data=train,
        classes=CLASSES,
        reference="none",
        availability={"premium": "premium_available"},
        weights="quote_weight",
    ).fit(alpha=0.25, compute_covariance=False)

    holdout_probabilities = result.predict_proba(holdout)
    holdout_loss = weighted_log_loss(holdout, holdout_probabilities)
    base_mix = result.tier_mix(holdout, weights="quote_weight")
    top_two = result.predict_top_k(holdout.head(5), k=2)

    calibration = result.fit_calibration(holdout)
    calibrated_mix = result.tier_mix(holdout, weights="quote_weight", calibration=calibration)

    scenario = result.scenario(
        holdout,
        changes={"price_premium": 1.03},
        weights="quote_weight",
        value_columns={
            "basic": "price_basic",
            "standard": "price_standard",
            "premium": "price_premium",
        },
        categorical_factors=["Channel"],
    )
    diagnostics = result.diagnostics(
        train_data=train,
        test_data=holdout,
        categorical_factors=["Channel"],
        continuous_factors=["DriverAge"],
    )

    print(result.summary())
    print(f"Held-out weighted log loss: {holdout_loss:.4f}")
    print(f"Holdout tier mix: {format_mix(base_mix)}")
    print(f"Calibrated holdout mix: {format_mix(calibrated_mix)}")
    print(f"Premium price +3% tier mix: {format_mix(scenario.scenario_class_mix)}")
    print(f"Premium mix delta: {scenario.class_mix_delta['premium']:.2%}")
    if scenario.expected_value is not None:
        print(f"Expected premium delta: {scenario.expected_value['delta']:.2f}")
    print(f"Train/test log-loss gap: {diagnostics.train_test_comparison['log_loss_delta']:.4f}")
    print("Top two classes for first five holdout quotes:")
    print(top_two)


if __name__ == "__main__":
    main()
