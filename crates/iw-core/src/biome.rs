use serde::{Deserialize, Serialize};

/// The 14 WWF terrestrial biomes plus water/ice markers (DESIGN.md §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Biome {
    /// Not yet classified (pre-biome phases).
    Unclassified,
    TropicalMoistBroadleaf,
    TropicalDryBroadleaf,
    TropicalConifer,
    TemperateBroadleaf,
    TemperateConifer,
    BorealTaiga,
    TropicalGrassland,
    TemperateGrassland,
    FloodedGrassland,
    MontaneGrassland,
    Tundra,
    Mediterranean,
    Desert,
    Mangrove,
    Ocean,
    Lake,
    IceSheet,
}

impl Biome {
    pub const TERRESTRIAL: [Biome; 14] = [
        Biome::TropicalMoistBroadleaf,
        Biome::TropicalDryBroadleaf,
        Biome::TropicalConifer,
        Biome::TemperateBroadleaf,
        Biome::TemperateConifer,
        Biome::BorealTaiga,
        Biome::TropicalGrassland,
        Biome::TemperateGrassland,
        Biome::FloodedGrassland,
        Biome::MontaneGrassland,
        Biome::Tundra,
        Biome::Mediterranean,
        Biome::Desert,
        Biome::Mangrove,
    ];

    pub fn is_land(self) -> bool {
        !matches!(self, Biome::Ocean | Biome::Lake | Biome::Unclassified)
    }

    pub fn name(self) -> &'static str {
        match self {
            Biome::Unclassified => "Unclassified",
            Biome::TropicalMoistBroadleaf => "Tropical & Subtropical Moist Broadleaf Forest",
            Biome::TropicalDryBroadleaf => "Tropical & Subtropical Dry Broadleaf Forest",
            Biome::TropicalConifer => "Tropical & Subtropical Coniferous Forest",
            Biome::TemperateBroadleaf => "Temperate Broadleaf & Mixed Forest",
            Biome::TemperateConifer => "Temperate Coniferous Forest",
            Biome::BorealTaiga => "Boreal Forest / Taiga",
            Biome::TropicalGrassland => "Tropical & Subtropical Grassland & Savanna",
            Biome::TemperateGrassland => "Temperate Grassland & Shrubland",
            Biome::FloodedGrassland => "Flooded Grassland & Savanna",
            Biome::MontaneGrassland => "Montane Grassland & Shrubland",
            Biome::Tundra => "Tundra",
            Biome::Mediterranean => "Mediterranean Forest & Scrub",
            Biome::Desert => "Desert & Xeric Shrubland",
            Biome::Mangrove => "Mangrove",
            Biome::Ocean => "Ocean",
            Biome::Lake => "Lake",
            Biome::IceSheet => "Ice Sheet",
        }
    }
}
