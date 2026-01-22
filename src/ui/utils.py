import sys
import os


def resource_path(relative_path):
    """Retorna o caminho absoluto para recursos (imagens, estilos, binários)"""
    try:
        # PyInstaller cria uma pasta temporária em _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def setup_environment():
    """Configura o PATH do sistema e variáveis de ambiente para o FFmpeg"""
    # 1. Garante que a pasta 'src' esteja no sys.path para importações de módulos irmãos
    # O arquivo atual está em src/ui/utils.py -> subimos dois níveis para achar a raiz do projeto/src
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.dirname(current_dir)  # src/

    if src_path not in sys.path:
        sys.path.append(src_path)

    # 2. Configuração do FFmpeg
    ffmpeg_path = resource_path(os.path.join("bin", "ffmpeg.exe"))

    if os.path.exists(ffmpeg_path):
        # Adiciona ao PATH para bibliotecas como pydub
        os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
        # Configura especificamente para imageio/moviepy
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
        print(f"✅ FFmpeg configurado internamente: {ffmpeg_path}")
    else:
        print("⚠️ FFmpeg não encontrado na pasta 'bin'. Usando instalação global.")
