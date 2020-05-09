package perceptron

type Minimizer func([]float64, float64) float64

func (fn Minimizer) Minimize(x []float64, yTrue float64) float64 {
	return fn(x, yTrue)
}
