use eframe::{egui, epi};
use eframe::epi::Frame;
use eframe::egui::{CtxRef, Color32, Label, Vec2, Layout, Ui, Grid};
use egui::plot::{Plot, Curve, Value};
use egui::widgets::{TextEdit};

use diffy_queue::runge_kutta::{RK4Solver, DOPRISolver};
use diffy_queue::euler::EulerSolver;
use diffy_queue::solver::Solver;
use std::f64::consts::E;
use std::error::Error;
use eframe::egui::Event::Text;

struct ComparativeSolver<'a> {
    initial_pt: (f64, f64),
    step_size: f64,
    euler_solver: EulerSolver<'a>,
    rk4_solver: RK4Solver<'a>,
    dopri_solver: DOPRISolver<'a>,
    actual_soln: Option<Box<dyn Fn(f64) -> f64 + 'a>>,

    dopri_step_bounds: (f64, f64),
}

impl<'a> ComparativeSolver<'a> {
    pub fn new() -> Self {
        let expr = "t*x".parse::<meval::Expr>().unwrap();

        ComparativeSolver {
            initial_pt: (0.0, 1.0),
            step_size: 0.01,
            euler_solver: EulerSolver::new(expr.clone().bind2("t", "x").unwrap(), 0.0, 1.0, 0.01),
            rk4_solver: RK4Solver::new(expr.clone().bind2("t", "x").unwrap(), 0.0, 1.0, 0.01),
            dopri_solver: DOPRISolver::new(expr.bind2("t", "x").unwrap(), 0.0, 1.0, 0.001, 0.0001, 0.001),
            actual_soln: Some(Box::new(|x: f64| E.powf(0.5 * x.powf(2.0)))),

            dopri_step_bounds: (0.0001, 0.001),
        }
    }

    pub fn set_eqn(&mut self, s: &String) -> Result<(), meval::Error> {
        let l = s.len();
        if l >= 2 && s.chars().nth(l-1).unwrap() == '^' && s.chars().nth(l-2).unwrap() == 'x' {
            println!("short circuiting");
            return Ok(());
        }
        println!("continuing");

        let expr = s.parse::<meval::Expr>()?;

        println!("{:?}", expr);
        self.euler_solver.set_fn(expr.clone().bind2("t", "x")?);
        self.dopri_solver.set_fn(expr.clone().bind2("t", "x")?);
        self.rk4_solver.set_fn(expr.bind2("t", "x")?);
        Ok(())
    }

    pub fn set_actual_soln(&mut self, s: &String) -> Result<(), meval::Error> {
        if *s != "" {
            self.actual_soln = Some(Box::new(s.parse::<meval::Expr>()?.bind("t")?));
        } else {
            self.actual_soln = None;
        }
        Ok(())
    }

    pub fn set_initial_t(&mut self, t: &String) -> Result<(), Box<dyn Error>>{
        self.initial_pt.0 = t.parse()?;
        self.euler_solver.set_initial_pt(self.initial_pt);
        self.dopri_solver.set_initial_pt(self.initial_pt);
        self.rk4_solver.set_initial_pt(self.initial_pt);
        Ok(())
    }

    pub fn set_initial_x(&mut self, x: &String) -> Result<(), Box<dyn Error>> {
        self.initial_pt.1 = x.parse()?;
        self.euler_solver.set_initial_pt(self.initial_pt);
        self.dopri_solver.set_initial_pt(self.initial_pt);
        self.rk4_solver.set_initial_pt(self.initial_pt);
        Ok(())
    }

    pub fn set_step_size(&mut self, h: &String) -> Result<(), Box<dyn Error>> {
        self.step_size = h.parse()?;
        println!("Resetting");
        self.euler_solver.set_step_size(self.step_size);
        self.dopri_solver.set_step_size(self.step_size);
        self.rk4_solver.set_step_size(self.step_size);
        Ok(())
    }

    pub fn set_dopri_min(&mut self, min: &String) -> Result<(), Box<dyn Error>> {
        self.dopri_step_bounds.0 = min.parse()?;
        self.dopri_solver.set_step_bounds(self.dopri_step_bounds);
        Ok(())
    }

    pub fn set_dopri_max(&mut self, max: &String) -> Result<(), Box<dyn  Error>> {
        self.dopri_step_bounds.1 = max.parse()?;
        self.dopri_solver.set_step_bounds(self.dopri_step_bounds);
        Ok(())
    }
}

#[derive(PartialEq)]
enum SolverView {
    Grapher,
    Table
}


pub struct SolverApp<'a> {
    eqn_str: String,
    eqn_str_color: Color32,
    actual_soln_str: String,
    actual_soln_str_color: Color32,
    step_size_str: String,
    step_size_str_color: Color32,
    t_max_str: String,
    t_max: i64,

    dopri_bounds_str: (String, String),
    dopri_bounds_color: (Color32, Color32),

    initial_pt_str: (String, String),
    initial_pt_color: (Color32, Color32),

    selected_view: SolverView,

    table_points_str: String,
    table_points_color: Color32,
    table_points: Vec<f64>,

    solvers: ComparativeSolver<'a>,
}

impl<'a> SolverApp<'a> {
    pub fn new() -> Self {
        SolverApp {
            eqn_str: "t*x".to_string(),
            eqn_str_color: Color32::WHITE,

            actual_soln_str: "e^(0.5*t^2)".to_string(),
            actual_soln_str_color: Color32::WHITE,

            initial_pt_str: ("0.0".to_string(), "1.0".to_string()),
            initial_pt_color: (Color32::WHITE, Color32::WHITE),

            step_size_str: "0.01".to_string(),
            step_size_str_color: Color32::WHITE,

            t_max_str: "3".to_string(),
            t_max: 3,

            dopri_bounds_str: ("0.0001".to_string(), "0.001".to_string()),
            dopri_bounds_color: (Color32::WHITE, Color32::WHITE),

            selected_view: SolverView::Grapher,
            table_points: vec![],
            table_points_str: String::new(),
            table_points_color: Color32::WHITE,

            solvers: ComparativeSolver::new(),
        }
    }

    fn render_header(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.with_layout(Layout::left_to_right(), |ui| ui.heading("Differential Equation Solver"));
            ui.with_layout(Layout::right_to_left(), |ui| {
                ui.radio_value(&mut self.selected_view, SolverView::Grapher, "Grapher");
                ui.radio_value(&mut self.selected_view, SolverView::Table, "Table");
            });
        });
        ui.horizontal(|ui| {
            ui.add(Label::new("Solver Parameters: ").strong());
            ui.label("dx/dt =");
            if ui.add(TextEdit::singleline(&mut self.eqn_str)
                .text_color(self.eqn_str_color).desired_width(BIG_WIDTH))
                .on_hover_text("An equation of x and/or t to solve as a differential. Note \
                    that all operations must be explicitly marked -- 'tx' must be written as 't*x'")
                .changed() {
                self.eqn_str_color = match self.solvers.set_eqn(&self.eqn_str) {
                    Ok(()) => Color32::WHITE,
                    Err(_) => Color32::RED
                };
            }
            ui.label("x(t) =");
            if ui.add(TextEdit::singleline(&mut self.actual_soln_str)
                .text_color(self.actual_soln_str_color).desired_width(BIG_WIDTH))
                .on_hover_text("The actual solution to the given differential (optional), \
                    to visually compare accuracy of the numerical solutions").changed() {
                self.actual_soln_str_color = match self.solvers.set_actual_soln(&self.actual_soln_str) {
                    Ok(()) => Color32::WHITE,
                    Err(_) => Color32::RED,
                };
            }

            ui.label("t_0 =");
            if ui.add(TextEdit::singleline(&mut self.initial_pt_str.0)
                .text_color(self.initial_pt_color.0).desired_width(SMALL_WIDTH))
                .on_hover_text("The initial t-value to solve with").changed() {
                self.initial_pt_color.0 = match self.solvers.set_initial_t(&self.initial_pt_str.0) {
                    Ok(()) => Color32::WHITE,
                    Err(_) => Color32::RED,
                };
            }
            ui.label("x_0 =");
            if ui.add(TextEdit::singleline(&mut self.initial_pt_str.1)
                .text_color(self.initial_pt_color.1).desired_width(SMALL_WIDTH))
                .on_hover_text("The initial x-value to solve with").changed() {
                self.initial_pt_color.1 = match self.solvers.set_initial_x(&self.initial_pt_str.1) {
                    Ok(()) => Color32::WHITE,
                    Err(_) => Color32::RED,
                };
            }
            ui.label("h =");
            if ui.add(TextEdit::singleline(&mut self.step_size_str)
                .text_color(self.step_size_str_color).desired_width(SMALL_WIDTH))
                .on_hover_text("The step size to solve with").changed() {
                self.step_size_str_color = match self.solvers.set_step_size(&self.step_size_str) {
                    Ok(()) => Color32::WHITE,
                    Err(_) => Color32::RED,
                }
            }
        });

        ui.horizontal(|ui| {
            ui.add(Label::new("DOPRI Parameters: ").strong());
            ui.label("h_min =");
            if ui.add(TextEdit::singleline(&mut self.dopri_bounds_str.0)
                .desired_width(SMALL_WIDTH))
                .on_hover_text("Minimum step size for Dormand-Prince solver").changed() {
                self.dopri_bounds_color.0 = match self.solvers.set_dopri_min(&self.dopri_bounds_str.0) {
                    Ok(()) => Color32::WHITE,
                    Err(_) => Color32::RED,
                }
            }
            ui.label("h_max =");
            if ui.add(TextEdit::singleline(&mut self.dopri_bounds_str.1)
                .desired_width(SMALL_WIDTH))
                .on_hover_text("Maximum step size for Dormand-Prince solver").changed() {
                self.dopri_bounds_color.1 = match self.solvers.set_dopri_max(&self.dopri_bounds_str.1) {
                    Ok(()) => Color32::WHITE,
                    Err(_) => Color32::RED,
                }
            }
        });
    }

    fn render_graph_view(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.add(Label::new("Graph Parameters: ").strong());
            ui.label("t_max =");
            if ui.add(TextEdit::singleline(&mut self.t_max_str)
                .desired_width(SMALL_WIDTH))
                .on_hover_text("The maximum t-value to solve up to and display").changed() {
                match self.t_max_str.parse::<i64>() {
                    Ok(t) => self.t_max = t,
                    Err(_) => (),
                }
            }
        });
        ui.separator();

        ui.vertical_centered(|ui| {
            ui.columns(2, |cols| {
                let mut soln_plot = Plot::new("Solution Plot")
                    .curve(Curve::from_values_iter(((self.solvers.initial_pt.0 as i64 * 100)..=self.t_max * 100)
                        .map(|x| x as f64 / 100.0)
                        .map(|x| Value::new(x, self.solvers.euler_solver.solve_at_point(x).unwrap()))
                    ).name("Euler Solution"))
                    .curve(Curve::from_values_iter(((self.solvers.initial_pt.0 as i64 * 100)..=self.t_max * 100)
                        .map(|x| x as f64 / 100.0)
                        .map(|x| Value::new(x, self.solvers.rk4_solver.solve_at_point(x).unwrap()))
                    ).name("RK4 Solution"))
                    .curve(Curve::from_values_iter(((self.solvers.initial_pt.0 as i64 * 100)..=self.t_max * 100)
                        .map(|x| x as f64 / 100.0)
                        .map(|x| Value::new(x, self.solvers.dopri_solver.solve_at_point(x).unwrap()))
                    ).name("DOPRI Solution"));
                if self.solvers.actual_soln.is_some() {
                    soln_plot = soln_plot.curve(Curve::from_values_iter(((self.solvers.initial_pt.0 as i64 * 100)..=self.t_max * 100)
                        .map(|x| x as f64 / 100.0)
                        .map(|x| Value::new(x, (self.solvers.actual_soln.as_ref().unwrap())(x)))
                    ).name("Actual Solution"));//.height(ui.available_size().y / 2.);
                }

                cols[0].heading("Solution Plot: ");
                cols[0].add(soln_plot);

                let mut err_plot = Plot::new("Error Plot");
                if self.solvers.actual_soln.is_some() {
                        err_plot = err_plot.curve(Curve::from_values_iter(
                            self.solvers.euler_solver.solved_pts.iter().filter(|p| p.0 < self.t_max as f64).map(
                                |pt| Value::new(pt.0, pt.1 - self.solvers.actual_soln.as_ref().unwrap()(pt.0))
                            )
                        ).name("Euler Error"))
                        .curve(Curve::from_values_iter(
                            self.solvers.rk4_solver.solved_pts.iter().filter(|p| p.0 < self.t_max as f64).map(
                                |pt| Value::new(pt.0, pt.1 - self.solvers.actual_soln.as_ref().unwrap()(pt.0))
                            )
                        ).name("RK4 Error"))
                        .curve(Curve::from_values_iter(
                            self.solvers.dopri_solver.solved_pts.iter().filter(|p| p.0 < self.t_max as f64).map(
                                |pt| Value::new(pt.0, pt.1 - self.solvers.actual_soln.as_ref().unwrap()(pt.0))
                            )
                        ).name("DOPRI Error"))
                }
                cols[1].heading("Error Plot: ");
                cols[1].add(err_plot);
            });
        });
    }

    fn render_table_view(&mut self, ui: &mut Ui) {
        ui.vertical_centered(|ui| {
            ui.label("Enter a comma-separated list of points to solve at: ");
            if ui.add(TextEdit::multiline(&mut self.table_points_str)
                .text_color(self.table_points_color))
                .changed() {
                self.table_points_color = match self._parse_table_points() {
                    Ok(_) => Color32::WHITE,
                    Err(_) => Color32::RED,
                }
            }

            Grid::new("points_table").striped(true).show(ui, |ui| {
                ui.label("t");
                ui.label("x(t)");
                ui.label("Euler");
                ui.label("RK4");
                ui.label("DOPRI");
                ui.end_row();

                for pt in &self.table_points {
                    ui.label(format!("{}", pt));
                    if self.solvers.actual_soln.is_some() {
                        ui.label(format!("{}", self.solvers.actual_soln.as_ref().unwrap()(*pt) ));
                    } else {
                        ui.label("n/a");
                    }
                    ui.label(format!("{}", self.solvers.euler_solver.solve_at_point(*pt).unwrap()));
                    ui.label(format!("{}", self.solvers.rk4_solver.solve_at_point(*pt).unwrap()));
                    ui.label(format!("{}", self.solvers.dopri_solver.solve_at_point(*pt).unwrap()));
                    ui.end_row();
                }
            });
        });
    }

    #[inline(always)]
    fn _parse_table_points(&mut self) -> Result<(), Box<dyn Error>> {
        let mut points_str = self.table_points_str.clone();
        points_str.retain(|x| !x.is_whitespace());
        self.table_points.clear();
        for pt in points_str.split(",") {
            self.table_points.push(pt.parse()?);
        }

        Ok(())
    }
}

const SMALL_WIDTH: f32 = 75.0;
const BIG_WIDTH: f32 = 125.0;
impl epi::App for SolverApp<'_> {
    fn update(&mut self, ctx: &CtxRef, _frame: &mut Frame<'_>) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_header(ui);

            match self.selected_view {
                SolverView::Grapher => self.render_graph_view(ui),
                SolverView::Table => self.render_table_view(ui),
            }
        });
    }

    fn name(&self) -> &str {
        "Grapher App"
    }
}
