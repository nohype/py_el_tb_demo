import logging
import os
import subprocess
import venv
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s - %(levelname)s] %(message)s")


def create_venv(dir: Path, clear_existing: bool = False) -> bool:
    if dir.exists():
        if not clear_existing:
            logging.warning(
                f"Skipping existing venv path {dir}{' - Note: It\'s not a directory!' if not dir.is_dir() else ''}"
            )
            return dir.is_dir()
        elif clear_existing and (not dir.is_dir()):
            logging.error(f"Cannot clear existing venv path {dir}: It's not a directory!")
            return False

    try:
        venv.EnvBuilder(clear=True, with_pip=True, upgrade_deps=True).create(dir)
        logging.info(f"Created venv at {dir}")
        return True
    except Exception as e:
        logging.error(f"Failed to create venv at {dir}: {e}")
        return False


def install_packages(venv_dir: Path, requirements_file: Path) -> bool:
    try:
        pip_path = venv_dir / "Scripts" / "pip.exe"

        if not pip_path.exists():
            logging.error(f"pip not found at {pip_path}")
            return False

        if not requirements_file.exists():
            logging.error(f"Requirements file not found at {requirements_file}")
            return False

        result = subprocess.run(
            [str(pip_path), "install", "-r", str(requirements_file)],
        )

        if result.returncode != 0:
            logging.error(f"Failed to install packages")
            return False

        logging.info(f"Successfully installed packages from {requirements_file}")
        return True
    except Exception as e:
        logging.error(f"Failed to install packages: {e}")
        return False


def regenerate_el_parser(antlr_jar_file: Path, antlr_grammar_file: Path, output_dir: Path) -> bool:
    try:
        if not antlr_jar_file.exists():
            logging.error(f"ANTLR jar file not found at {antlr_jar_file}")
            return False

        if not antlr_grammar_file.exists():
            logging.error(f"ANTLR grammar file not found at {antlr_grammar_file}")
            return False

        if output_dir.exists() and not output_dir.is_dir():
            logging.error(f"Output path exists but is not a directory: {output_dir}")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                "java",
                "-jar",
                str(antlr_jar_file),
                "-visitor",
                "-no-listener",
                str(antlr_grammar_file),
                "-o",
                str(output_dir),
                "-encoding",
                "utf-8",
                "-package",
                "elantlr",
                "-Dlanguage=Python3",
            ],
            cwd=output_dir,
            text=True,
        )

        if result.returncode != 0:
            logging.error(f"Failed to generate EL parser: {result.stderr}")
            return False

        logging.info(f"Successfully generated EL parser at {output_dir}")
        return True
    except Exception as e:
        logging.error(f"Failed to regenerate EL parser: {e}")
        return False


def main():
    venv_path = Path(__file__).parent / ".venv"
    requirements_file = Path(__file__).parent / "requirements.txt"
    antlr_jar_file = Path(__file__).parent / "external/el_grammar/vendor/ANTLR4/antlr-4.13.2-complete.jar"
    antlr_grammar_file = Path(__file__).parent / "external/el_grammar/grammar/EasyLanguage.g4"
    antlr_output_dir = Path(__file__).parent / "generated/elantlr"

    if not create_venv(dir=venv_path, clear_existing=True):
        exit(1)

    if not install_packages(venv_dir=venv_path, requirements_file=requirements_file):
        exit(1)

    if not regenerate_el_parser(
        antlr_jar_file=antlr_jar_file, antlr_grammar_file=antlr_grammar_file, output_dir=antlr_output_dir
    ):
        exit(1)


if __name__ == "__main__":
    main()
