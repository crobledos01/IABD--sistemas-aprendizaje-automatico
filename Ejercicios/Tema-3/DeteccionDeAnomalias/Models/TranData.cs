using Microsoft.ML.Data;

namespace DeteccionDeAnomalias.Models;

public class TranData
{
    [LoadColumn(0)]
    public float idTran { get; set; }

    [LoadColumn(1)]
    public float Date { get; set; }

    [LoadColumn(2)]
    public float Import { get; set; }

    [LoadColumn(3)]
    public float NumTrans { get; set; }

    [LoadColumn(2)]
    public float LogImport { get; set; }

    [LoadColumn(3)]
    public float ImportByTran { get; set; }

}

public class TranPrediction : TranData
{
    // true = anomalía
    public bool PredictedLabel { get; set; }

    // Cuanto mayor, más anómalo
    // Por defecto >= 0.5 anomalía
    public float Score { get; set; }
}