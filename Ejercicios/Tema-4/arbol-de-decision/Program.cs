var estudiantes = new[] {
    new { estudia = true, duermeBien = true, entregaTareas = true, aprueba = true },
    new { estudia = true, duermeBien = false, entregaTareas = true, aprueba = true },
    new { estudia = false, duermeBien = true, entregaTareas = true, aprueba = true },
    new { estudia = false, duermeBien = false, entregaTareas = false, aprueba = false },
    new { estudia = true, duermeBien = true, entregaTareas = false, aprueba = true },
    new { estudia = false, duermeBien = false, entregaTareas = true, aprueba = false }
};

// Las listas sirven para clasificar a los estudiantes según el resultado del árbol de decisión
// El error se utiliza para verificar que no haya estudiantes mal clasificados,
//en caso de que el árbol no fuese correcto saltaría un error al correr el código

var aprobados = new List<dynamic>();
var suspensos = new List<dynamic>();
var error = false;

foreach (var estudiante in estudiantes)
{
    if (estudiante.estudia)
    {
        aprobados.Add(estudiante);
        if(!estudiante.aprueba)
        {
            error = true;
            System.Console.WriteLine("Hay un estudiante que no aprueba en un if final de aprobado");
        }
    }
    else
    {
        if (estudiante.duermeBien)
        {
            aprobados.Add(estudiante);
        
            if(!estudiante.aprueba)
            {
                error = true;
                System.Console.WriteLine("Hay un estudiante que no aprueba en un if final de aprobado");
            }
        }
        else
        {
            suspensos.Add(estudiante);
        
            if(estudiante.aprueba)
            {
                error = true;
                System.Console.WriteLine("Hay un estudiante que no aprueba en un if final de suspenso");
            }
        }
    }
}

if (aprobados.Count + suspensos.Count == estudiantes.Length && !error)
{
    System.Console.WriteLine("Todos los estudiantes del primer grupo han sido clasificados correctamente.");
}

// Segunda prueba con duermeBien como primera pregunta para comprobar que se resuelve de la misma forma

var aprobados2 = new List<dynamic>();
var suspensos2 = new List<dynamic>();
var error2 = false;

foreach (var estudiante in estudiantes)
{
    if (estudiante.duermeBien)
    {
        aprobados2.Add(estudiante);
        
        if(!estudiante.aprueba)
        {
            error2 = true;
            System.Console.WriteLine("Hay un estudiante que no aprueba en un if final de aprobado");
        }
    }
    else
    {
        if (estudiante.estudia)
        {
            aprobados2.Add(estudiante);
        
            if(!estudiante.aprueba)
            {
                error2 = true;
                System.Console.WriteLine("Hay un estudiante que no aprueba en un if final de aprobado");
            }
        }
        else
        {
            suspensos2.Add(estudiante);
        
            if(estudiante.aprueba)
            {
                error2 = true;
                System.Console.WriteLine("Hay un estudiante que no aprueba en un if final de suspenso");
            }
        }
    }
}


if (aprobados2.Count() + suspensos2.Count() == estudiantes.Length && !error2)
{
    System.Console.WriteLine("Todos los estudiantes del segundo grupo han sido clasificados correctamente.");
}