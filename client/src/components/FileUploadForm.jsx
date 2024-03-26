import React, { useState } from "react";
import ResultsDisplay from "./ResultsDisplay";

const FileUploadForm = () => {
    const [file, setFile] = useState(null);
    const [results, setResults] = useState(null);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!file) {
            alert("Please upload a file first.");
            return;
        }

        setTimeout(() => setResults("Processed Data"), 2000);
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input type="file" onChange={handleFileChange} />
                <button type="submit">Process Data</button>
            </form>

            {results && <ResultsDisplay results={results} />}
        </div>
    );
};

export default FileUploadForm;
