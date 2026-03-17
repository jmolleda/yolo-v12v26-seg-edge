import subprocess
import sys

def main():
    scripts_entrenamiento = [
        "nano/train26nBenchMark.py",
        "small/train26sBenchMark.py",
        "medium/train26mBenchMark.py",
        "large/train26LBenchMark.py"
    ]

    print("🚀 Iniciando la cola de entrenamientos YOLO para el TFM...\n")

    for script in scripts_entrenamiento:
        print(f"{'='*50}")
        print(f"⏳ Ejecutando modelo: {script}")
        print(f"{'='*50}")

        try:
            subprocess.run(["python3", script], check=True)
            print(f"\n{script} finalizado con éxito.\n")
            
        except subprocess.CalledProcessError as e:
            print(f"\nError durante la ejecución de {script}.")
            print(f"Código de salida: {e.returncode}")
            print("Deteniendo la cola de entrenamiento para que puedas revisar el error.")
            sys.exit(1)
            
        except FileNotFoundError:
            print(f"\nNo se pudo encontrar el archivo '{script}'. Verifica el nombre y la ruta.")
            sys.exit(1)

    print("Todos los modelos han terminado de entrenarse.")

if __name__ == "__main__":
    main()