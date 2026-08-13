//! The cell inspector: everything the current snapshot knows about one cell,
//! plus its stratigraphic column drawn as a stacked bar.
//!
//! The column is not part of `PlanetView` (64 strata per cell is far too much
//! to snapshot at 4 Hz), so it is fetched from the simulation worker with
//! [`iw_sim::SimHandle::request_column`] and cached until the selection or the
//! snapshot version changes.

use iw_core::{planet::cell_flags, Biome, CrustType, MetamorphicGrade, PlanetView, Stratum};
use iw_mesh::Mesh;
use iw_render_vulkan::egui;

use crate::layers;

/// Named tectonic flag bits set on a cell, in bit order.
pub fn flag_names(bits: u8) -> Vec<&'static str> {
    const NAMED: [(u8, &str); 7] = [
        (cell_flags::SUBDUCTING, "subducting"),
        (cell_flags::ARC, "volcanic arc"),
        (cell_flags::COLLISION, "collision"),
        (cell_flags::RIFT, "rift / ridge"),
        (cell_flags::HOTSPOT, "hotspot"),
        (cell_flags::TRANSFORM, "transform"),
        (cell_flags::SUTURE, "suture"),
    ];
    NAMED
        .iter()
        .filter(|(bit, _)| bits & bit != 0)
        .map(|(_, name)| *name)
        .collect()
}

/// Mass-weighted mean density of a column, kg/m^3. `PlanetView` does not carry
/// `Planet::crust_density_kg_m3`, so the inspector reports this instead.
pub fn column_density_kg_m3(column: &[Stratum]) -> Option<f32> {
    let total: f32 = column.iter().map(|s| s.thickness_m).sum();
    if total <= 0.0 {
        return None;
    }
    let mass: f32 = column
        .iter()
        .map(|s| s.thickness_m * s.rock.density_kg_m3())
        .sum();
    Some(mass / total)
}

/// Short name of a metamorphic grade.
pub fn grade_name(grade: MetamorphicGrade) -> &'static str {
    match grade {
        MetamorphicGrade::None => "unaltered",
        MetamorphicGrade::Low => "low grade",
        MetamorphicGrade::Medium => "medium grade",
        MetamorphicGrade::High => "high grade",
    }
}

/// Selection state and the cached column for it.
#[derive(Default)]
pub struct Inspector {
    /// Selected cell, if any.
    pub cell: Option<u32>,
    /// Cached strata, bottom first.
    pub column: Vec<Stratum>,
    /// Snapshot version the cached column was fetched at.
    pub column_version: u64,
    /// True while a column request is outstanding.
    pub awaiting_column: bool,
}

impl Inspector {
    /// Select a cell (dropping any cached column for the previous one).
    pub fn select(&mut self, cell: u32) {
        if self.cell != Some(cell) {
            self.column.clear();
            self.column_version = 0;
        }
        self.cell = Some(cell);
    }

    /// Forget the selection.
    pub fn clear(&mut self) {
        self.cell = None;
        self.column.clear();
        self.column_version = 0;
        self.awaiting_column = false;
    }

    /// Whether a fresh column should be requested for snapshot `version`.
    pub fn needs_column(&self, version: u64) -> bool {
        self.cell.is_some() && !self.awaiting_column && self.column_version != version
    }

    /// Store a column that arrived for snapshot `version`.
    pub fn set_column(&mut self, column: Vec<Stratum>, version: u64) {
        self.column = column;
        self.column_version = version;
        self.awaiting_column = false;
    }
}

/// Draw the inspector window. Returns false when the user closed it.
pub fn show(
    ctx: &egui::Context,
    inspector: &Inspector,
    mesh: &Mesh,
    view: &PlanetView,
    historic: bool,
) -> bool {
    let Some(cell) = inspector.cell else {
        return true;
    };
    let i = cell as usize;
    let cells = &view.cells;
    if i >= cells.elevation_m.len() {
        return true;
    }
    let mut open = true;
    egui::Window::new("Cell inspector")
        .open(&mut open)
        .default_pos([1180.0, 40.0])
        // Tall enough that the stratigraphic column is on screen without a
        // scroll or a resize; egui clamps this to the available space.
        .default_size([380.0, 780.0])
        .vscroll(true)
        .show(ctx, |ui| {
            let ll = mesh.latlon[i];
            ui.label(format!(
                "cell {cell}  {}  {}",
                format_lat(ll[0]),
                format_lon(ll[1])
            ));
            if historic {
                ui.colored_label(
                    egui::Color32::from_rgb(0xd0, 0xa0, 0x40),
                    "scrubbing history - fields below are from the live snapshot",
                );
            }
            ui.separator();

            let elev = cells.elevation_m[i];
            let above = elev - view.sea_level_m;
            grid(ui, "cell-basics", |ui| {
                row(ui, "elevation", format!("{elev:.0} m"));
                row(
                    ui,
                    "above sea",
                    format!(
                        "{above:.0} m ({})",
                        if above >= 0.0 { "land" } else { "submerged" }
                    ),
                );
                row(ui, "sea level", format!("{:.0} m", view.sea_level_m));
                row(ui, "area", format!("{:.0} km2", mesh.areas_km2[i]));
                let plate = cells.plate_id[i];
                row(
                    ui,
                    "plate",
                    if plate == layers::NO_PLATE {
                        "unassigned".to_string()
                    } else {
                        let v = cells.plate_velocity_m_yr[i].length() * 100.0;
                        format!("#{plate}  {v:.1} cm/yr")
                    },
                );
                row(
                    ui,
                    "crust",
                    match cells.crust_type[i] {
                        CrustType::Oceanic => "oceanic".to_string(),
                        CrustType::Continental => "continental".to_string(),
                    },
                );
                row(
                    ui,
                    "crust thickness",
                    format!("{:.1} km", cells.crust_thickness_m[i] / 1000.0),
                );
                if cells.crust_type[i] == CrustType::Oceanic {
                    row(
                        ui,
                        "crust age",
                        format!("{:.1} Myr", cells.crust_age_myr[i]),
                    );
                }
                if let Some(d) = column_density_kg_m3(&inspector.column) {
                    row(ui, "column density", format!("{d:.0} kg/m3"));
                }
                row(
                    ui,
                    "temperature",
                    format!("{:.1} C", cells.temperature_c[i]),
                );
                row(
                    ui,
                    "precipitation",
                    format!("{:.0} mm/yr", cells.precip_mm_yr[i]),
                );
                row(ui, "ice", format!("{:.0} m", cells.ice_thickness_m[i]));
                row(ui, "water flux", format_flux(cells.water_flux_m3_yr[i]));
                row(ui, "lake depth", format!("{:.0} m", cells.lake_depth_m[i]));
                row(ui, "sediment", format!("{:.0} m", cells.sediment_m[i]));
                row(ui, "biome", biome_label(cells.biome[i]).to_string());
            });

            let flags = flag_names(cells.tectonic_flags[i]);
            ui.horizontal_wrapped(|ui| {
                ui.label("tectonics:");
                if flags.is_empty() {
                    ui.weak("quiet");
                } else {
                    ui.label(flags.join(", "));
                }
            });

            ui.separator();
            ui.heading("Stratigraphic column");
            if inspector.column.is_empty() {
                ui.weak(if inspector.awaiting_column {
                    "asking the simulation..."
                } else {
                    "no strata deposited here yet (bare mantle)"
                });
            } else {
                draw_column(ui, &inspector.column);
            }
        });
    open
}

/// Vertical stacked bar of the strata, top of the column at the top of the
/// bar, each stratum's height proportional to its thickness.
fn draw_column(ui: &mut egui::Ui, column: &[Stratum]) {
    let total: f32 = column.iter().map(|s| s.thickness_m).sum();
    ui.label(format!("{} strata, {:.0} m total", column.len(), total));
    let bar_w = 46.0;
    let bar_h = 240.0;
    ui.horizontal_top(|ui| {
        let (rect, _response) =
            ui.allocate_exact_size(egui::vec2(bar_w, bar_h), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, egui::Color32::from_gray(20));
        let mut y = rect.top();
        // Top of the column first: strata are stored bottom-first.
        for (idx, stratum) in column.iter().enumerate().rev() {
            let frac = if total > 0.0 {
                stratum.thickness_m / total
            } else {
                0.0
            };
            let h = (frac * bar_h).max(1.0);
            let seg = egui::Rect::from_min_size(
                egui::pos2(rect.left(), y),
                egui::vec2(bar_w, h.min(rect.bottom() - y)),
            );
            let c = layers::rock_color(Some(stratum.rock));
            painter.rect_filled(seg, 0.0, egui::Color32::from_rgb(c[0], c[1], c[2]));
            if stratum.grade != MetamorphicGrade::None {
                // Hatch metamorphosed layers with a bright edge so the grade is
                // visible even in a one-pixel band.
                painter.line_segment(
                    [seg.left_top(), seg.right_top()],
                    egui::Stroke::new(1.0, egui::Color32::from_gray(230)),
                );
            }
            ui.interact(seg, ui.id().with(("stratum", idx)), egui::Sense::hover())
                .on_hover_text(stratum_line(stratum));
            y += h;
            if y >= rect.bottom() {
                break;
            }
        }
        painter.rect_stroke(
            rect,
            2.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(90)),
            egui::StrokeKind::Inside,
        );

        ui.vertical(|ui| {
            egui::ScrollArea::vertical()
                .max_height(bar_h)
                .show(ui, |ui| {
                    for stratum in column.iter().rev() {
                        ui.horizontal(|ui| {
                            let c = layers::rock_color(Some(stratum.rock));
                            let (r, _) = ui
                                .allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                            ui.painter().rect_filled(
                                r,
                                1.0,
                                egui::Color32::from_rgb(c[0], c[1], c[2]),
                            );
                            ui.label(stratum_line(stratum));
                        });
                    }
                });
        });
    });
}

/// One line of strata detail: rock, thickness, deposition age, grade.
pub fn stratum_line(s: &Stratum) -> String {
    format!(
        "{:?}  {:.0} m  @{:.1} Myr  {}",
        s.rock,
        s.thickness_m,
        s.deposited_myr,
        grade_name(s.grade)
    )
}

fn grid(ui: &mut egui::Ui, id: &str, f: impl FnOnce(&mut egui::Ui)) {
    egui::Grid::new(id)
        .num_columns(2)
        .spacing([10.0, 2.0])
        .show(ui, f);
}

fn row(ui: &mut egui::Ui, label: &str, value: String) {
    ui.weak(label);
    ui.label(value);
    ui.end_row();
}

fn biome_label(b: Biome) -> &'static str {
    b.name()
}

/// Discharge in readable units (m^3/yr up to km^3/yr).
pub fn format_flux(m3_yr: f32) -> String {
    if m3_yr >= 1.0e9 {
        format!("{:.2} km3/yr", m3_yr / 1.0e9)
    } else if m3_yr > 0.0 {
        format!("{m3_yr:.3e} m3/yr")
    } else {
        "none".to_string()
    }
}

/// Latitude as degrees with a hemisphere letter.
pub fn format_lat(lat_rad: f32) -> String {
    let d = lat_rad.to_degrees();
    format!("{:.2} {}", d.abs(), if d >= 0.0 { "N" } else { "S" })
}

/// Longitude as degrees with a hemisphere letter.
pub fn format_lon(lon_rad: f32) -> String {
    let d = lon_rad.to_degrees();
    format!("{:.2} {}", d.abs(), if d >= 0.0 { "E" } else { "W" })
}

#[cfg(test)]
mod tests {
    use super::*;
    use iw_core::RockType;

    #[test]
    fn flags_are_named_in_bit_order() {
        assert!(flag_names(0).is_empty());
        assert_eq!(flag_names(cell_flags::RIFT), vec!["rift / ridge"]);
        assert_eq!(
            flag_names(cell_flags::ARC | cell_flags::SUBDUCTING),
            vec!["subducting", "volcanic arc"]
        );
        assert_eq!(flag_names(0xff).len(), 7, "one name per defined bit");
    }

    #[test]
    fn column_density_is_mass_weighted() {
        let s = |rock, thickness_m| Stratum {
            rock,
            thickness_m,
            deposited_myr: 0.0,
            grade: MetamorphicGrade::None,
        };
        assert_eq!(column_density_kg_m3(&[]), None);
        let one = column_density_kg_m3(&[s(RockType::Granite, 100.0)]).unwrap();
        assert!((one - RockType::Granite.density_kg_m3()).abs() < 1e-3);
        // Thick basalt under thin granite must pull the mean toward basalt.
        let mixed =
            column_density_kg_m3(&[s(RockType::Basalt, 900.0), s(RockType::Granite, 100.0)])
                .unwrap();
        let expect = (900.0 * RockType::Basalt.density_kg_m3()
            + 100.0 * RockType::Granite.density_kg_m3())
            / 1000.0;
        assert!((mixed - expect).abs() < 1e-2);
    }

    #[test]
    fn selection_invalidates_the_cached_column() {
        let mut ins = Inspector::default();
        assert!(!ins.needs_column(1), "nothing selected");
        ins.select(7);
        assert!(ins.needs_column(1));
        ins.awaiting_column = true;
        assert!(!ins.needs_column(1), "one request at a time");
        ins.set_column(vec![], 1);
        assert!(!ins.needs_column(1), "already have version 1");
        assert!(ins.needs_column(2), "a new snapshot refetches");
        ins.select(7);
        assert!(!ins.needs_column(1), "reselecting the same cell keeps it");
        ins.select(8);
        assert!(ins.column.is_empty() && ins.needs_column(1));
        ins.clear();
        assert!(ins.cell.is_none());
    }

    #[test]
    fn formatting_is_readable() {
        assert_eq!(format_lat(0.0), "0.00 N");
        assert!(format_lat(-0.5).ends_with('S'));
        assert!(format_lon(-1.0).ends_with('W'));
        assert_eq!(format_flux(0.0), "none");
        assert!(format_flux(2.0e9).starts_with("2.00 km3"));
        assert!(format_flux(5.0).contains("m3/yr"));
        let line = stratum_line(&Stratum {
            rock: RockType::Shale,
            thickness_m: 120.0,
            deposited_myr: 33.25,
            grade: MetamorphicGrade::Low,
        });
        assert!(line.contains("Shale") && line.contains("120 m") && line.contains("low grade"));
    }
}
