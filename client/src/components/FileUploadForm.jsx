import React, { useState } from "react";
import axios from "axios";
import ResultsDisplay from "./ResultsDisplay";

const FileUploadForm = () => {
    const [testData, setTestData] = useState(null);
    const [testLabel, setTestLabel] = useState(null);
    const [results, setResults] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    const handleTestDataChange = (event) => {
        setTestData(event.target.files[0]);
    };

    const handleTestLabelChange = (event) => {
        setTestLabel(event.target.files[0]);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!testData || !testLabel) {
            alert("Please upload both data and label files.");
            return;
        }

        const formData = new FormData();
        formData.append("testData", testData);
        formData.append("testLabel", testLabel);

        setIsLoading(true);
        try {
            const response = await axios.post(
                "http://localhost:3000/analyze-eeg",
                formData,
                {
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                }
            );
            setResults(response.data.inferenceResult);
        } catch (error) {
            console.error("Error uploading files:", error);
            alert("Failed to process data");
        }
        setIsLoading(false);
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <label htmlFor="testData">Upload Test Data: </label>
                <input
                    id="testData"
                    type="file"
                    onChange={handleTestDataChange}
                />
                <label htmlFor="testLabel">Upload Test Labels: </label>
                <input
                    id="testLabel"
                    type="file"
                    onChange={handleTestLabelChange}
                />
                <div>
                    <button type="submit" disabled={isLoading}>
                        {isLoading ? "Processing..." : "Process Data"}
                    </button>
                </div>
            </form>

            {results && <ResultsDisplay results={results} />}
        </div>
    );
};

export default FileUploadForm;
