
pub struct LinearRegression {
    slope: f64,
    intercept: f64,
}

impl LinearRegression {
    /// Ajusta uma linha aos dados usando regressão linear simples
    pub fn fit(data: &[f64]) -> Self {
        let n = data.len() as f64;
        let x_mean = (0..data.len()).map(|x| x as f64).sum::<f64>() / n;
        let y_mean = data.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, y) in data.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        let slope = numerator / denominator;
        let intercept = y_mean - slope * x_mean;

        Self { slope, intercept }
    }

    /// Faz uma previsão com base na equação da reta
    pub fn predict(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }

    /// Faz previsões para `future_steps` períodos à frente
    pub fn predict_future(&self, last_index: usize, future_steps: usize) -> Vec<f64> {
        (1..=future_steps)
            .map(|i| self.predict((last_index + i) as f64))
            .collect()
    }

    /// Calcula o erro quadrático médio (MSE)
    pub fn mean_squared_error(&self, data: &[f64]) -> f64 {
        let mse: f64 = data
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let y_pred = self.predict(i as f64);
                (y - y_pred).powi(2)
            })
            .sum();
        mse / data.len() as f64
    }

    /// Calcula o coeficiente de determinação (R²)
    pub fn r_squared(&self, data: &[f64]) -> f64 {
        let mean_y = data.iter().sum::<f64>() / data.len() as f64;
        let ss_total: f64 = data.iter().map(|y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = data
            .iter()
            .enumerate()
            .map(|(i, y)| (y - self.predict(i as f64)).powi(2))
            .sum();
        1.0 - (ss_res / ss_total)
    }

    /// Retorna os coeficientes (slope e intercept)
    pub fn coefficients(&self) -> (f64, f64) {
        (self.slope, self.intercept)
    }
}

fn main() {
    let data = vec![2.0, 4.1, 6.0, 8.2, 10.1]; // série temporal

    let model = LinearRegression::fit(&data);
    let (slope, intercept) = model.coefficients();

    println!("Slope: {:.4}, Intercept: {:.4}", slope, intercept);
    println!("R²: {:.4}", model.r_squared(&data));
    println!("MSE: {:.4}", model.mean_squared_error(&data));

    let future = model.predict_future(data.len() - 1, 3);
    println!("Próximos 3 valores previstos: {:?}", future);
}
