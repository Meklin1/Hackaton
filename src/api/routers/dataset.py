from fastapi import APIRouter, File, UploadFile, Depends
from fastapi.responses import Response
from src.api.models.responses import SuccessResponse, DatasetDataResponse, ErrorResponse
from pathlib import Path


router = APIRouter(
    prefix="/dataset",
    tags=["dataset"],
)


def get_dataset_info() -> dict:
    """
    Returns a dictionary of available datasets in the 'data' directory.
    Format:
    {
        "0": {
            "name": <dataset_name_without_extension>,
            "path": <Path object to model file>
        },
        ...
    }
    """
    # Get the project root directory (where this file is located, go up to project root)
    project_root = Path(__file__).parent.parent.parent.parent
    datasets_dir = project_root / "data"

    if not datasets_dir.exists():
        return {}

    try:
        datasets = [data for data in datasets_dir.iterdir() if data.suffix == ".csv"]
        return {
            str(idx): {"name": dataset.stem, "path": dataset}
            for idx, dataset in enumerate(datasets)
        }
    except Exception as e:
        print(f"Error reading dataset directory: {e}")
        return {}


@router.get("/get-datasets")
async def get_datasets(datasets_info: dict = Depends(get_dataset_info)):
    return DatasetDataResponse(
        message="Datasets fetched successfully", data=datasets_info
    )


@router.get(f"/get-dataset/{dataset_id}")
async def get_dataset(dataset_id: str, datasets_info: dict = Depends(get_dataset_info)):
    
    csv_output = datasets_info.get(dataset_id).get("path").read_csv(index=False)
    # Return as CSV file download
    return Response(
        content=csv_output,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=exoplanet_predictions.csv"
        },
    )


@router.post("/upload-dataset")
async def upload_dataset(
    csv_file: UploadFile = File(...), datasets_info: dict = Depends(get_dataset_info)
):
    try:
        # Get the project root directory and create the data directory path
        project_root = Path(__file__).parent.parent.parent.parent
        data_dir = project_root / "data"

        # Ensure the data directory exists
        data_dir.mkdir(exist_ok=True)

        csv_data = await csv_file.read()
        file_path = data_dir / csv_file.filename
        with open(file_path, "wb") as f:
            f.write(csv_data)
    except Exception as e:
        return ErrorResponse(
            error=f"Error Saving CSV file :( {csv_file.filename}", detail=str(e)
        )

    return SuccessResponse(message="Datasets Uploaded Successfully", data=[])
