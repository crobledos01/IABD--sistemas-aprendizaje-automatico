using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Linq;

class Program
{
    //////////////////////////////////////////////
    /// IMPUTAR DATOS
    //////////////////////////////////////////////

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static double Mean(IEnumerable<double> values) => values.Average();

    static double Median(IEnumerable<double> values)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    static (double value, int count) Mode(IEnumerable<double> values, int decimals = 2)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    static string ModeCategorical(IEnumerable<string> values)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    //////////////////////////////////////////////
    /// ESCALAR DATOS NUMERICOS
    //////////////////////////////////////////////

    static double[] MinMax(double[] values)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    static double[] ZScore(double[] values)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    //////////////////////////////////////////////
    /// CODIFICAR VARIABLES CATEGORICAS
    //////////////////////////////////////////////

    static Dictionary<string, int> CreateLabelEncoder(IEnumerable<string> values)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    static (string header, int[] vector, Dictionary<string, int> encoding) LabelEncoding(string[] values, string colName)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    static (string[] headers, int[][] matrix, Dictionary<string, int> encoding) OneHotEncoding(string[] values, string colName)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    //////////////////////////////////////////////
    /// FICHEROS
    //////////////////////////////////////////////

    static List<string[]> ReadCsv(string path)
    {
        var rows = new List<string[]>();
        foreach (var line in System.IO.File.ReadLines(path))
        {
            rows.Add(line.Split(',')); // Cambia el separador si tu archivo usa otro
        }
        return rows;
    }

    static void WriteCsv(string path, List<string[]> outRows)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    //////////////////////////////////////////////
    /// UTILIDADES
    //////////////////////////////////////////////

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static string Safe(string value) => value?.Trim() ?? string.Empty;

    static double? ToNullableDouble(string value)
    {
        value = Safe(value);
        if (string.IsNullOrEmpty(value) || value.Equals("NaN", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        if (double.TryParse(value, NumberStyles.Any, CultureInfo.InvariantCulture, out double v))
            return v;

        if (double.TryParse(value, NumberStyles.Any, new CultureInfo("es-ES"), out v))
            return v;

        return null;
    }

    static string StringToDouble(double v) => v.ToString("0.######", CultureInfo.InvariantCulture);

    //////////////////////////////////////////////
    /// PROGRAMA PRINCIPAL
    //////////////////////////////////////////////

    static void Main()
    {
        const string fileInputPath = "data.csv";
        const string fileOutputPath = "data_preprocessed.csv";

        var rows = ReadCsv(fileInputPath);
        if (rows.Count < 2)
        {
            Console.WriteLine("CSV vacio o sin datos.");
            return;
        }

        var header = rows[0];
        var data = rows.Skip(1).ToList();

        // Mapeo de nombre de columna a indice
        var colIndexMap = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < header.Length; i++)
        {
            colIndexMap[header[i]] = i;
        }

        string[] headerColumnNames = {
            "ID", "Edad", "Genero", "Ingresos_Mensuales", "Gastos_Anuales", "Educacion", "Calificacion_Credito", "Tiempo_Empleo",
        };

        // TODO: Implementar comprobacion

        string[] numericColumnNames = {
            "Edad", "Ingresos_Mensuales", "Gastos_Anuales", "Calificacion_Credito", "Tiempo_Empleo",
        };

        // TODO: Implementar

        string[] categoricalColumnNames = {
            "Genero", "Educacion",
        };

        // TODO: Implementar

        var scaledCols = new Dictionary<string, double[]>();
        // TODO: Implementar

        string[] generoVals = data.Select(row => Safe(row[colIndexMap["Genero"]])).ToArray();
        // var (genHeaders, genOheMatrix, genEncoder) = // TODO: Implementar

        string[] eduVals = data.Select(row => Safe(row[colIndexMap["Educacion"]])).ToArray();
        // var (eduHeaders, eduOheMatrix, eduEncoder) = // TODO: Implementar

        double[] ratio = data.Select(row =>
        {
            // TODO: Implementar
            return 0.0;
        }).ToArray();

        var outHeader = new List<string>();
        // outHeader.Add("ID");
        // outHeader.AddRange(numericColumnNames);
        // outHeader.AddRange(genHeaders);
        // outHeader.AddRange(eduHeaders);
        // outHeader.Add("Ratio_Deuda");
        // outHeader.AddRange(scaledCols.Keys);

        var outRows = new List<string[]>
        {
            outHeader.ToArray()
        };

        for (int i = 0; i < data.Count; i++)
        {
            var row = new List<string>();
            // TODO: ID
            // TODO: numericas imputadas
            // TODO: OHE genero
            // TODO: OHE educacion
            // TODO: ratio
            // TODO: escaladas
            outRows.Add(row.ToArray());
        }

        // TODO: WriteCsv

        Console.WriteLine($"OK -> {fileOutputPath}");
    }
}
