import json
import re
from pathlib import Path

from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

# auto проверяет и скачивает подходящие ноутбуки
# manual скачивает выбранный ноутбук
# review проверяет сохраненные ноутбуки
MODE = "auto"

NOTEBOOK = "dmitryalexandrovski/titanic-from-baseline-to-statistical-validation"
COMPETITION = "bike-sharing-demand"
LIMIT = 50
DOWNLOAD_CSV = True
MIN_VOTES = None
SORT_BY = "voteCount"
MIN_ANALYSIS_CELLS = 3
EXCLUDED_NOTEBOOKS = [
    "muhammadaslamjaved/logistic-regression-the-mathematics-of-titanic",
    "muhammadaslamjaved/knn-from-scratch-learning-by-similarity",
    "muhammadaslamjaved/decision-trees-the-mathematics-of-splitting",
    "danielsowah123/titanic-survival-prediction",
    "mjasimrizwan/titanic-eda-survival-analysis-with-pandas-seaborn",
    "dahmanimarouamanar/house-prices-prediction",
    "ismailuzum/house-prices-pytorch-mlp",
    "roshaaann30/house-price-prediction-with-xgboost",
    "golu1234567jat/house-price-prediction-feature-engineering",
    "sheharyaaarr/house-price-xg-boost",
    "aliderakhshesh/bike-sharing-a-simple-start-point",
    "fatmakursun/bike-sharing-feature-engineering",
    "homayoonkhadivi/2-creative-deep-learning-models-bike-prediction"
]

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

hypothesis_words = (
    "hypoth", "i suspect", "we suspect", "we expect", "we assume",
    "may indicate", "suggests a potential", "i would expect",
    "гипотез", "предполага", "ожида",
    "test whether", "evaluate whether", "determine whether",
    "examine whether", "question was raised",
    "can lead", "could improve", "could reduce",
    "should improve", "should reduce", "might explain", "проверим"
)

conclusion_words = (
    "conclusion", "insight", "finding", "observation",
    "**answer", "confirmed", "rejected", "we found", "we observed",
    "the results show", "improved", "improvement", "worsened",
    "we find", "we can see", "we can conclude", "we see",
    "this indicates", "this suggests", "shows that",
    "the chart reveals", "this chart", "supports our hypothesis",
    "findings of", "performance gain", "вывод", "заключени",
    "performs", "cut the error", "подтверд", "отверг", "получили",
    "улучшил", "ухудш", "показал"
)

output_words = (
    "rmse", "p-value", "p_value", "f-stat", "t-stat", "accuracy",
    "cv ", "score", "improvement", "ошибка"
)

unfinished_words = (
    "(fill in)", "[fill in]", "**fill in**", "fill in.",
    "write your own conclusion", "to adapt with your own numbers",
    "заполните вывод", "впишите результат"
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
    result_paragraphs = 0
    errors = 0
    hypothesis_fragments = []
    result_fragments = []
    unfinished_conclusions = []
    conclusion_level = 0

    for number, cell in enumerate(notebook["cells"], start=1):
        source = "".join(cell["source"])
        content = source.lower()

        if cell["cell_type"] == "markdown":
            text.append(cell)
            if len(content.strip()) > 100:
                explanations += 1
                if any(word in content for word in conclusion_words):
                    conclusions += 1

            for line in source.splitlines():
                line = line.strip()
                if line.startswith("#"):
                    level = len(line) - len(line.lstrip("#"))
                    if any(word in line.lower() for word in ("conclusion", "вывод", "заключени", "итог")):
                        conclusion_level = level
                    elif conclusion_level and level <= conclusion_level:
                        conclusion_level = 0
                if conclusion_level and any(word in line.lower() for word in unfinished_words):
                    unfinished_conclusions.append({"cell": number, "text": line[:600]})

            clean_source = re.sub(r"<[^>]*>", " ", source)
            for paragraph in clean_source.split("\n\n"):
                paragraph = paragraph.strip()
                if len(paragraph) < 40:
                    continue
                lower = paragraph.lower()
                if len(hypothesis_fragments) < 3 and any(word in lower for word in hypothesis_words):
                    hypothesis_fragments.append({"cell": number, "text": paragraph[:600]})
                if analysis > 0 and (not check_outputs or outputs > 0) and any(word in lower for word in conclusion_words):
                    if not any(word in lower for word in ("whether", "we will", "objective", "future", "could", "might", "let's test")):
                        result_paragraphs += 1
                        result_fragments.append({"cell": number, "kind": "markdown", "text": paragraph[:600]})
                        result_fragments = result_fragments[-3:]

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
                output_text = "".join(output.get("text", []))
                if not output_text:
                    output_text = "".join(output.get("data", {}).get("text/plain", []))
                if output_text.strip() and ("score_dataset(" in content or any(word in output_text.lower() for word in output_words)):
                    result_fragments.append({"cell": number, "kind": "output", "text": output_text[:600]})
                    result_fragments = result_fragments[-3:]

    language = get_language(text)
    candidate = (
        language in ("ru", "en")
        and explanations >= 2
        and analysis >= MIN_ANALYSIS_CELLS
        and data_analysis > 0
        and result_paragraphs > 0
        and not unfinished_conclusions
    )
    if check_outputs and (outputs == 0 or errors > 0):
        candidate = False
    if MIN_VOTES is not None and (votes is None or votes < MIN_VOTES):
        candidate = False

    return {
        "text_language": language,
        "code_cells": code_cells,
        "text_cells": len(text),
        "explanation_cells": explanations,
        "analysis_cells": analysis,
        "data_analysis_cells": data_analysis,
        "conclusion_cells": conclusions,
        "output_cells": outputs,
        "result_paragraphs": result_paragraphs,
        "error_cells": errors,
        "hypothesis_fragments": hypothesis_fragments,
        "result_fragments": result_fragments,
        "unfinished_conclusions": unfinished_conclusions,
        "votes": votes,
        "candidate": candidate
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
        if MODE == "auto" and notebook in EXCLUDED_NOTEBOOKS:
            continue

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
        except (OSError, ValueError, KeyError, TypeError):
            review = {"candidate": False}

        reference = path.parent.parent.name + "/" + path.parent.name
        if reference in EXCLUDED_NOTEBOOKS:
            review["candidate"] = False

        item = {"notebook": str(path)}
        item.update(review)
        report.append(item)

    report.sort(key=lambda item: (item["candidate"], item.get("votes") or 0), reverse=True)
    save_json(Path("data/review.json"), report)
