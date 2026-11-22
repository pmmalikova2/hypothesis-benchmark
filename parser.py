
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiDownloadKernelOutputRequest
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
from kagglesdk.competitions.types.competition_api_service import ApiGetCompetitionRequest
from langdetect import detect, LangDetectException

# auto проверяет и скачивает подходящие ноутбуки
# manual скачивает выбранный ноутбук
# review проверяет сохраненные ноутбуки
MODE = "review"

NOTEBOOK = "dmitryalexandrovski/titanic-from-baseline-to-statistical-validation"
COMPETITION = "titanic"
LIMIT = 20
DOWNLOAD_CSV = True

analysis_words = (
    "describe(", "groupby(", "value_counts(",
    "corr(", "plot(", "hist(", "sns.",
    "plt.", "pivot_table(", "scatter("
)


# сохраняет json
def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


# получает список файлов
def get_files(name, kind):
    files = []
    token = None

    while True:
        if kind == "dataset":
            page = api.dataset_list_files(name, page_token=token)
        else:
            page = api.competition_list_files(name, page_token=token)

        for item in page.files or []:
            files.append({
                "name": item.name,
                "format": Path(item.name).suffix,
                "size": item.total_bytes
            })

        token = page.next_page_token

        if not token:
            return files


# определяет язык текста
def get_language(cells):
    text = " ".join("".join(cell["source"]) for cell in cells)

    if len(text.strip()) < 100:
        return "unknown"

    try:
        language = detect(text)
    except LangDetectException:
        return "unknown"

    return language if language in ("ru", "en") else "other"


# скачивает csv если их еще нет
def download_csv(name, kind, files, folder):
    if not DOWNLOAD_CSV:
        return

    for item in files:
        if item["format"].lower() != ".csv":
            continue

        if (folder / item["name"]).exists():
            continue

        if kind == "dataset":
            api.dataset_download_file(
                name, item["name"], path=str(folder), unzip=True
            )
        else:
            api.competition_download_file(
                name, item["name"], path=str(folder), unzip=True
            )


if MODE != "review":
    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()

    if MODE == "manual":
        notebooks = [NOTEBOOK]
    else:
        results = api.kernels_list(
            competition=COMPETITION,
            language="python",
            kernel_type="notebook",
            page_size=LIMIT
        )
        notebooks = [item.ref for item in results or []]

    for notebook in notebooks:
        owner, name = notebook.split("/")
        folder = Path("data/notebooks") / COMPETITION / owner / name

        request = ApiGetKernelRequest()
        request.user_name = owner
        request.kernel_slug = name

        response = client.kernels.kernels_api_client.get_kernel(request)
        info = response.metadata

        # проверяет текст и анализ до скачивания результатов
        if MODE == "auto":
            source = json.loads(response.blob.source)

            text = [
                cell for cell in source["cells"]
                if cell["cell_type"] == "markdown"
            ]

            code = [
                "".join(cell["source"]).lower()
                for cell in source["cells"]
                if cell["cell_type"] == "code"
            ]

            explanations = sum(
                len("".join(cell["source"]).strip()) > 100
                for cell in text
            )

            analysis = sum(
                any(word in content for word in analysis_words)
                for content in code
            )

            if get_language(text) not in ("ru", "en") or explanations < 2 or analysis == 0:
                continue

        # скачивает выполненный ноутбук
        request = ApiDownloadKernelOutputRequest()
        request.owner_slug = owner
        request.kernel_slug = name
        request.version_number = info.current_version_number
        request.file_path = "__notebook__.ipynb"

        with client.kernels.kernels_api_client.download_kernel_output(request) as response:
            response.raise_for_status()
            content = response.content

        if "cells" not in json.loads(content):
            raise ValueError("Скачанный файл не является ноутбуком")

        folder.mkdir(parents=True, exist_ok=True)
        (folder / (name + ".ipynb")).write_bytes(content)

        save_json(folder / "metadata.json", {
            "id": info.id,
            "title": info.title,
            "author": info.author,
            "url": "https://www.kaggle.com/code/" + notebook,
            "language": info.language,
            "type": info.kernel_type,
            "version": info.current_version_number,
            "votes": info.total_votes,
            "datasets": info.dataset_data_sources or [],
            "competitions": info.competition_data_sources or []
        })

        for dataset in info.dataset_data_sources or []:
            dataset_owner, dataset_name = dataset.split("/")
            dataset_folder = Path("data/datasets") / dataset_owner / dataset_name

            request = ApiGetDatasetRequest()
            request.owner_slug = dataset_owner
            request.dataset_slug = dataset_name

            data = client.datasets.dataset_api_client.get_dataset(request)
            files = get_files(dataset, "dataset")

            save_json(dataset_folder / "metadata.json", {
                "id": data.id,
                "title": data.title,
                "url": "https://www.kaggle.com/datasets/" + dataset,
                "description": data.description,
                "size": data.total_bytes,
                "license": data.license_name,
                "files": files
            })

            download_csv(dataset, "dataset", files, dataset_folder)

    request = ApiGetCompetitionRequest()
    request.competition_name = COMPETITION
    competition = client.competitions.competition_api_client.get_competition(request)

    folder = Path("data/competitions") / COMPETITION
    files = get_files(COMPETITION, "competition")

    save_json(folder / "metadata.json", {
        "id": competition.id,
        "title": competition.title,
        "url": competition.url,
        "description": competition.description,
        "files": files
    })

    download_csv(COMPETITION, "competition", files, folder)


if MODE in ("auto", "review"):
    conclusion_words = (
        "conclusion", "insight", "finding",
        "observation", "hypothesis", "вывод"
    )

    report = []

    for path in Path("data/notebooks").rglob("*.ipynb"):
        with open(path) as file:
            notebook = json.load(file)

        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        text = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]

        code_text = ["".join(cell["source"]).lower() for cell in code]
        markdown_text = ["".join(cell["source"]).lower() for cell in text]

        explanations = sum(len(content.strip()) > 100 for content in markdown_text)
        analysis = sum(any(word in content for word in analysis_words) for content in code_text)
        conclusions = sum(any(word in content for word in conclusion_words) for content in markdown_text)
        outputs = sum(bool(cell.get("outputs")) for cell in code)
        language = get_language(text)

        report.append({
            "notebook": str(path),
            "text_language": language,
            "code_cells": len(code),
            "text_cells": len(text),
            "explanation_cells": explanations,
            "analysis_cells": analysis,
            "conclusion_cells": conclusions,
            "output_cells": outputs,
            "candidate": language in ("ru", "en") and explanations >= 2 and analysis >= 1
        })

    save_json(Path("data/review.json"), report)
