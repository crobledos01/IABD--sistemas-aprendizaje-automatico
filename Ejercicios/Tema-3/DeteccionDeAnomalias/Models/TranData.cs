using Microsoft.ML.Data;

namespace DeteccionDeAnomalias.Models;

public class TranData
{
    [LoadColumn(0)]
    public float idTran { get; set; }

    [LoadColumn(1)]
    public float idUser { get; set; }

    [LoadColumn(2)]
    public float Date { get; set; }

    [LoadColumn(3)]
    public float Hour { get; set; }

    [LoadColumn(4)]
    public float Import { get; set; }

    [LoadColumn(5)]
    public float Country { get; set; }

    [LoadColumn(6)]
    public float Type { get; set; }

    [LoadColumn(7)]
    public float NumTrans { get; set; }

    [LoadColumn(8)]
    public float PayMethod { get; set; }

}

public class TranPrediction : TranData
{
    // true = anomalía
    public bool PredictedLabel { get; set; }

    // Cuanto mayor, más anómalo
    // Por defecto >= 0.5 anomalía
    public float Score { get; set; }
}