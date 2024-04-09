import React, { useState } from "react";
import axios from "axios";
import ResultsDisplay from "./ResultsDisplay";

const FileUploadForm = () => {
    const [file, setFile] = useState(null);
    const [results, setResults] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!file) {
            alert("Please upload a file first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

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
            console.error("Error uploading file:", error);
            alert("Failed to process data");
        }
        setIsLoading(false);
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input type="file" onChange={handleFileChange} />
                <button type="submit" disabled={isLoading}>
                    {isLoading ? "Processing..." : "Process Data"}
                </button>
            </form>

            {results && <ResultsDisplay results={results} />}
        </div>
    );
};

export default FileUploadForm;
