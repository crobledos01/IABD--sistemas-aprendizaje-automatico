using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Linq;
using System.IO;

class Program
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static double Mean(IEnumerable<double> values) => values.Average();

    static double Median(IEnumerable<double> values)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    static (double value, int count) Mode(IEnumerable<double> values, int decimals = 2)
    {
        var round_values = values.Select(v => Math.Round(v, decimals));
        var groups = round_values.GroupBy(x => x).OrderByDescending(g => g.Count());
        var first = groups.FirstOrDefault();
        return first != null ? (first.Key, first.Count()) : (0.0, 0);
    }

    static string ModeCategorical(IEnumerable<string> values)
    {
        var valid = values.Where(v => !string.IsNullOrWhiteSpace(v));
        var groups = valid.GroupBy(x => x).OrderByDescending(g => g.Count());
        return groups.FirstOrDefault()?.Key ?? "";
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
            rows.Add(line.Split(',')); 
        }
        return rows;
    }

    static void WriteCsv(string path, List<string[]> outRows)
    {
        using (var writer = new StreamWriter(path, false))
        {
            foreach (var row in outRows)
            {
                writer.WriteLine(string.Join(",", row));
            }
        }
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

        var num_modes = new Dictionary<string, double>();
        foreach (var column_name in numericColumnNames)
        {
            int id = colIndexMap[column_name];
            var nums = data.Select(row => ToNullableDouble(row[id])).Where(x => x.HasValue).Select(x => x.Value);
            var mode = Mode(nums).value;
            num_modes[column_name] = mode;
        }

        string[] categoricalColumnNames = {
            "Genero", "Educacion",
        };

        var cat_modes = new Dictionary<string, string>();
        foreach (var column_name in categoricalColumnNames)
        {
            int idx = colIndexMap[column_name];
            var vals = data.Select(row => Safe(row[idx]));
            cat_modes[column_name] = ModeCategorical(vals);
        }

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
        for (int j = 0; j < header.Length; j++)
        {
            outHeader.Add(header[j]);
        }

        var outRows = new List<string[]>
        {
            outHeader.ToArray()
        };

        for (int i = 0; i < data.Count; i++)
        {
            var row = new List<string>();
            for (int j = 0; j < header.Length; j++)
            {
                string column_name = header[j];
                string value = Safe(data[i][j]);
                if (num_modes.ContainsKey(column_name) && (string.IsNullOrEmpty(value) || value.Equals("NaN", StringComparison.OrdinalIgnoreCase)))
                    row.Add(StringToDouble(num_modes[column_name]));
                else if (cat_modes.ContainsKey(column_name) && string.IsNullOrEmpty(value))
                    row.Add(cat_modes[column_name]);
                else
                    row.Add(value);
            }
            outRows.Add(row.ToArray());
        }

        WriteCsv(fileOutputPath, outRows);

        Console.WriteLine($"OK -> {fileOutputPath}");
    }
}
