import requests
import datetime
import os
import platform
import yaml
from ultralytics import YOLO

NTFY_TOPIC = "yolo_uniovi_uo278095"

def enviar_notificacion(titulo, mensaje, tipo="info"):
    """
    Envía una notificación al canal de ntfy.sh
    """
    url = f"https://ntfy.sh/{NTFY_TOPIC}"

    if tipo == "error":
        headers = {
            "Title": titulo,
            "Priority": "urgent",
            "Tags": "warning,skull"
        }
    else:
        headers = {
            "Title": titulo,
            "Priority": "default",
            "Tags": "tada,partying_face"
        }

    try:
        requests.post(url, 
                      data=mensaje.encode('utf-8'), 
                      headers=headers)
        print(f"--> Notificación enviada al canal: {NTFY_TOPIC}")
    except Exception as e:
        print(f"--> No se pudo enviar la notificación (Error de red): {e}")


dir_script = os.path.dirname(os.path.abspath(__file__))

ruta_data_yaml = os.path.join(os.path.dirname(dir_script), "data", "data.yaml") 

ruta_config = os.path.join(os.path.dirname(dir_script), "hiperparametros.yaml")

try:
    with open(ruta_config, 'r') as archivo_yaml:
        params_compartidos = yaml.safe_load(archivo_yaml)
except FileNotFoundError:
    print(f"Error: No se encuentra el archivo {ruta_config}")
    exit(1)

train_params = {
    "data": ruta_data_yaml,
    "task": "segment",
}

train_params.update(params_compartidos)
train_params["lr0"] = 0.0005

versionYOLO = "versionesYolo/yolo26-seg-L.yaml"

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    machine_name = platform.node() 
    
    print(f"Inicio del script: {start_time.strftime('%H:%M:%S')}")
    print(f"Ejecutando en la máquina: {machine_name}")

    try:
        model = YOLO(versionYOLO) 

        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_guardado = os.path.join(directorio_actual, "resultados")

        results = model.train(
        project=ruta_guardado,
        name="modelo_large",
        **train_params
        )
        
        save_dir = results.save_dir

        print("\n--- Extrayendo métricas de inferencia para dispositivo de borde ---")
        val_results = model.val(data=train_params["data"], imgsz=train_params["imgsz"], split='val', project=save_dir,
        name="val")
        
        speed = val_results.speed
        t_pre = speed.get('preprocess', 0.0)
        t_inf = speed.get('inference', 0.0)
        t_post = speed.get('postprocess', 0.0)
        t_total_ms = t_pre + t_inf + t_post
        fps = 1000.0 / t_total_ms if t_total_ms > 0 else 0.0

        params_filepath = os.path.join(save_dir, "parametros_usados.txt")

        end_time = datetime.datetime.now()
        duration = end_time - start_time

        try:
            with open(params_filepath, 'w') as f:
                f.write("=== REPORTE DE ENTRENAMIENTO Y RENDIMIENTO ===\n")
                f.write(f"Máquina de Ejecución: {machine_name}\n")
                f.write(f"Modelo Base: {versionYOLO} (Random Weights)\n")
                f.write(f"Fecha de inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 50 + "\n")
                
                f.write("--- Rendimiento en Dispositivo (PyTorch) ---\n")
                f.write(f"Tiempo Pre-procesado : {t_pre:.2f} ms/img\n")
                f.write(f"Tiempo Inferencia    : {t_inf:.2f} ms/img\n")
                f.write(f"Tiempo Post-procesado: {t_post:.2f} ms/img (NMS & Mask)\n")
                f.write(f"Latencia Total       : {t_total_ms:.2f} ms/img\n")
                f.write(f"Velocidad Estimada   : {fps:.2f} FPS\n")
                f.write(f"Fecha de fin   : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duración total : {str(duration).split('.')[0]}\n")
                f.write("-" * 50 + "\n")
                
                f.write("--- Hiperparámetros de Entrenamiento ---\n")
                for key, value in train_params.items():
                    f.write(f"{key}: {value}\n")
        except Exception as e:
            print(f"Advertencia: No se pudo guardar el txt de parámetros: {e}")
        
        mensaje = (
            f"Máquina: {machine_name}\n"
            f"Modelo: {versionYOLO}\n"
            f"Duración: {str(duration).split('.')[0]}\n"
            f"Rendimiento: {fps:.1f} FPS | Latencia: {t_total_ms:.1f}ms\n"
            f"Guardado en: {save_dir}"
        )
        
        enviar_notificacion("Entrenamiento Completado", mensaje, tipo="info")

    except Exception as e:
        mensaje_error = f"Máquina: {machine_name}\nEl script se ha detenido por un error:\n{str(e)}"
        print(f"CRITICAL ERROR: {e}")
        enviar_notificacion("Fallo en el Entrenamiento", mensaje_error, tipo="error")
        raise e