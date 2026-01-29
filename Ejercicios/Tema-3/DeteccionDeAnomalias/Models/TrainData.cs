using Microsoft.ML.Data;

namespace DeteccionDeAnomalias.Models;

public class TrainData
{
    [LoadColumn(0)]
    public float indexTiempo { get; set; }

    [LoadColumn(1)]
    public float usoCPU { get; set; }

    [LoadColumn(2)]
    public float usoMemoria { get; set; }

    [LoadColumn(3)]
    public float velVent { get; set; }

    [LoadColumn(2)]
    public float temperatura { get; set; }

    [LoadColumn(3)]
    public float label { get; set; }

}

public class TrainPrediction : TrainData
{
    // true = anomalía
    public bool PredictedLabel { get; set; }

    // Cuanto mayor, más anómalo
    // Por defecto >= 0.5 anomalía
    public float Score { get; set; }
}