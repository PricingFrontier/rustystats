"""Tier-conversion multinomial example."""

import numpy as np
import polars as pl
import rustystats as rs


def make_quotes(n: int = 500, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(42.0, 12.0, size=n).clip(18.0, 80.0)
    vehicle_value = rng.lognormal(mean=10.0, sigma=0.35, size=n)
    channel = rng.choice(["direct", "agent", "partner"], p=[0.45, 0.35, 0.20], size=n)
    channel_agent = (channel == "agent").astype(float)
    channel_partner = (channel == "partner").astype(float)

    scaled_age = (age - 42.0) / 12.0
    scaled_value = (np.log(vehicle_value) - 10.0) / 0.35
    eta = np.column_stack(
        [
            np.zeros(n),
            -0.20 + 0.30 * scaled_age - 0.25 * scaled_value - 0.10 * channel_agent,
            -0.55 + 0.10 * scaled_age + 0.20 * scaled_value + 0.35 * channel_agent,
            -1.10 - 0.15 * scaled_age + 0.45 * scaled_value + 0.25 * channel_partner,
        ]
    )
    exp_eta = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs = exp_eta / exp_eta.sum(axis=1, keepdims=True)
    classes = np.array(["none", "basic", "standard", "premium"], dtype=object)
    purchased_tier = [rng.choice(classes, p=row) for row in probs]

    return pl.DataFrame(
        {
            "PurchasedTier": purchased_tier,
            "DriverAge": age,
            "VehicleValue": vehicle_value,
            "Channel": channel,
        }
    )


def main() -> None:
    quotes = make_quotes()
    terms = {
        "DriverAge": {"type": "bs", "df": 5},
        "VehicleValue": {"type": "linear"},
        "Channel": {"type": "categorical"},
    }

    result = rs.multinomial_dict(
        response="PurchasedTier",
        terms=terms,
        data=quotes,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(alpha=0.1, regularization="ridge", compute_covariance=False)

    base_mix = result.tier_mix(quotes)
    discount_offset = np.zeros((quotes.height, len(result.classes_)), dtype=np.float64)
    discount_offset[:, result.classes_.index("premium")] = 0.15
    scenario_mix = result.tier_mix(quotes, offset=discount_offset)

    print(result.summary())
    print("Base tier mix:", base_mix)
    print("Premium utility +0.15 scenario mix:", scenario_mix)


if __name__ == "__main__":
    main()
