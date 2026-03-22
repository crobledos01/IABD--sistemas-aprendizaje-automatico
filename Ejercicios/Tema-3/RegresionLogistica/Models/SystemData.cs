using Microsoft.ML.Data;
namespace RegresionLineal.Models;

public class SystemData
{
    [LoadColumn(0)]
    public float TempC { get; set; }

    [LoadColumn(1)]
    public float HumPct { get; set; }

    [LoadColumn(2)]
    public float PowerW { get; set; }

    [LoadColumn(3)]
    public float DeltaT { get; set; }

    [LoadColumn(4)]
    public float IsAnomaly { get; set; }

}