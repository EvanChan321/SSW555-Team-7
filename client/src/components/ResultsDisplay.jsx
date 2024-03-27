import React from "react";

const ResultsDisplay = ({ results }) => (
    <div>
        <h2>Results:</h2>
        <p>{getResult()}</p>
    </div>
);

const getResult = () => {
    let sim = Math.floor(Math.random() * 2);
    if(sim) return "True"
    return "False"
}

export default ResultsDisplay;
