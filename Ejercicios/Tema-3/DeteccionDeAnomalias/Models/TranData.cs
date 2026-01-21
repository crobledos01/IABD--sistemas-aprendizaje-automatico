using Microsoft.ML.Data;

namespace DeteccionDeAnomalias.Models;

public class TranData
{
    [LoadColumn(0)]
    public float idTran { get; set; }

    [LoadColumn(1)]
    public string idUser { get; set; }

    [LoadColumn(2)]
    public string Date { get; set; }

    [LoadColumn(3)]
    public string Hour { get; set; }

    [LoadColumn(4)]
    public float Import { get; set; }

    [LoadColumn(5)]
    public string Country { get; set; }

    [LoadColumn(6)]
    public string Type { get; set; }

    [LoadColumn(7)]
    public float NumTrans { get; set; }

    [LoadColumn(8)]
    public string PayMethod { get; set; }

}

public class TranPrediction : TranData
{
    // true = anomalía
    public bool PredictedLabel { get; set; }

    // Cuanto mayor, más anómalo
    // Por defecto >= 0.5 anomalía
    public float Score { get; set; }
}