//! Named world presets: parameter bundles for distinctly different planets.
//! A preset rewrites the physics knobs of a config in place, leaving the
//! seed, cell budget and storage settings alone so "same world, different
//! climate" comparisons stay easy.

use iw_core::PlanetConfig;

pub struct Preset {
    pub name: &'static str,
    pub blurb: &'static str,
    apply: fn(&mut PlanetConfig),
}

impl Preset {
    /// Rewrite `config`'s physics knobs; seed/budget/storage are preserved.
    pub fn apply(&self, config: &mut PlanetConfig) {
        let keep_seed = config.seed;
        let keep_budget = config.cell_budget;
        let keep_history = config.history_cap_bytes;
        *config = PlanetConfig::default();
        config.seed = keep_seed;
        config.cell_budget = keep_budget;
        config.history_cap_bytes = keep_history;
        (self.apply)(config);
    }
}

/// The preset table, Earth-like first (it is the default config verbatim).
pub const PRESETS: &[Preset] = &[
    Preset {
        name: "Earth-like",
        blurb: "the calibration baseline: ~27% land, temperate mix",
        apply: |_| {},
    },
    Preset {
        name: "Archipelago",
        blurb: "high seas over many small cratons: island chains everywhere",
        apply: |c| {
            c.water_budget = 1.25;
            c.craton_count = 24;
            c.tectonic_vigor = 1.3;
        },
    },
    Preset {
        name: "Pangaea",
        blurb: "sluggish mantle: the supercontinent barely breaks up",
        apply: |c| {
            c.tectonic_vigor = 0.45;
            c.water_budget = 0.88;
        },
    },
    Preset {
        name: "Desert world",
        blurb: "hot, dry, small seas: endless dunes and salt basins",
        apply: |c| {
            c.water_budget = 0.55;
            c.precip_multiplier = 0.45;
            c.temperature_offset_c = 4.0;
        },
    },
    Preset {
        name: "Ice age",
        blurb: "deep cold and strong glacial cycles: sheets reach mid-latitudes",
        apply: |c| {
            c.temperature_offset_c = -7.0;
            c.glacial_intensity = 1.8;
            c.water_budget = 0.9;
        },
    },
    Preset {
        name: "Hothouse jungle",
        blurb: "greenhouse heat and drenching rain: forest to the poles",
        apply: |c| {
            c.temperature_offset_c = 6.0;
            c.precip_multiplier = 1.8;
        },
    },
    Preset {
        name: "Waterworld",
        blurb: "nearly drowned: scattered volcanic islands in a world ocean",
        apply: |c| {
            c.water_budget = 1.9;
            c.hotspot_count = 24;
        },
    },
    Preset {
        name: "Volcanic",
        blurb: "restless mantle: hotspot chains, arcs and young mountains",
        apply: |c| {
            c.hotspot_count = 30;
            c.tectonic_vigor = 1.8;
            c.craton_count = 8;
        },
    },
];
