import json
from pathlib import Path

from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

# auto проверяет и скачивает подходящие ноутбуки
# manual скачивает выбранный ноутбук
# review проверяет сохраненные ноутбуки
MODE = "auto"

NOTEBOOK = "dmitryalexandrovski/titanic-from-baseline-to-statistical-validation"
COMPETITION = "titanic"
LIMIT = 20
DOWNLOAD_CSV = True
MIN_VOTES = None
SORT_BY = "hotness"
MIN_ANALYSIS_CELLS = 3

if MODE not in ("auto", "manual", "review"):
    raise ValueError("MODE должен быть auto, manual или review")

analysis_words = (
    "describe(", "groupby(", "value_counts(",
    "corr(", "plot(", "hist(", "barplot(",
    "boxplot(", "heatmap(", "countplot(",
    "pivot_table(", "crosstab(", "scatter("
)

data_analysis_words = (
    "describe(", "groupby(", "value_counts(",
    "corr(", "pivot_table(", "crosstab("
)

conclusion_words = (
    "conclusion", "insight", "finding", "observation", "hypothes",
    "answer", "suggest", "вывод", "гипотез", "наблюдени"
)


# сохраняет json
def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
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
    texts = ["".join(cell["source"]) for cell in cells]
    text = " ".join(texts)

    if len(text.strip()) < 100:
        return "unknown"

    try:
        language = detect(text)
    except LangDetectException:
        return "unknown"

    if language in ("ru", "en"):
        return language
    return "other"


def review_notebook(notebook, votes=None, check_outputs=True):
    text = []
    code_cells = 0
    explanations = 0
    analysis = 0
    data_analysis = 0
    conclusions = 0
    outputs = 0
    errors = 0

    for cell in notebook["cells"]:
        content = "".join(cell["source"]).lower()

        if cell["cell_type"] == "markdown":
            text.append(cell)
            if len(content.strip()) > 100:
                explanations += 1
                if any(word in content for word in conclusion_words):
                    conclusions += 1

        elif cell["cell_type"] == "code":
            code_cells += 1
            if any(word in content for word in analysis_words):
                analysis += 1
            if any(word in content for word in data_analysis_words):
                data_analysis += 1
            if cell.get("outputs"):
                outputs += 1
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    errors += 1
                    break

    language = get_language(text)

    reasons = []
    if language not in ("ru", "en"):
        reasons.append("Язык текста не русский и не английский")
    if explanations < 2:
        reasons.append("Меньше двух развернутых пояснений")
    if analysis < MIN_ANALYSIS_CELLS:
        reasons.append(f"Меньше {MIN_ANALYSIS_CELLS} ячеек анализа")
    if data_analysis < 1:
        reasons.append("Нет операций анализа таблицы: groupby, corr, describe и подобных")
    if conclusions < 1:
        reasons.append("Не найдены развернутые выводы или обсуждение результатов")
    if check_outputs and outputs == 0:
        reasons.append("Нет сохраненных результатов выполнения")
    if check_outputs and errors:
        reasons.append("В результатах выполнения есть ошибки")
    if MIN_VOTES is not None and (votes is None or votes < MIN_VOTES):
        reasons.append(f"Меньше {MIN_VOTES} голосов или число голосов неизвестно")

    return {
        "text_language": language,
        "code_cells": code_cells,
        "text_cells": len(text),
        "explanation_cells": explanations,
        "analysis_cells": analysis,
        "data_analysis_cells": data_analysis,
        "conclusion_cells": conclusions,
        "output_cells": outputs,
        "error_cells": errors,
        "votes": votes,
        "candidate": not reasons,
        "reasons": reasons
    }


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
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiDownloadKernelOutputRequest
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    from kagglesdk.competitions.types.competition_api_service import ApiGetCompetitionRequest

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
            page_size=LIMIT,
            sort_by=SORT_BY
        )
        notebooks = [item.ref for item in results or []]

    for notebook in notebooks:
        owner, name = notebook.split("/")
        folder = Path("data/notebooks") / COMPETITION / owner / name

        if MODE == "auto" and (folder / (name + ".ipynb")).exists():
            continue

        request = ApiGetKernelRequest()
        request.user_name = owner
        request.kernel_slug = name

        response = client.kernels.kernels_api_client.get_kernel(request)
        info = response.metadata

        if MODE == "auto":
            source = json.loads(response.blob.source)

            review = review_notebook(source, info.total_votes, check_outputs=False)
            if not review["candidate"]:
                continue

        request = ApiDownloadKernelOutputRequest()
        request.owner_slug = owner
        request.kernel_slug = name
        request.version_number = info.current_version_number
        request.file_path = "__notebook__.ipynb"

        with client.kernels.kernels_api_client.download_kernel_output(request) as response:
            response.raise_for_status()
            content = response.content

        downloaded = json.loads(content)
        if not isinstance(downloaded.get("cells"), list):
            raise ValueError("Скачанный файл не является ноутбуком")

        if MODE == "auto":
            review = review_notebook(downloaded, info.total_votes)
            if not review["candidate"]:
                continue

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
    report = []

    for path in sorted(Path("data/notebooks").rglob("*.ipynb")):
        try:
            with open(path, encoding="utf-8") as file:
                notebook = json.load(file)

            metadata_path = path.parent / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as file:
                    metadata = json.load(file)

            review = review_notebook(notebook, metadata.get("votes"))
        except (OSError, ValueError, KeyError, TypeError) as error:
            review = {"candidate": False, "reasons": [f"Не удалось проверить файл: {error}"]}

        item = {"notebook": str(path)}
        item.update(review)
        report.append(item)

    report.sort(key=lambda item: (item["candidate"], item.get("votes") or 0), reverse=True)
    save_json(Path("data/review.json"), report)
