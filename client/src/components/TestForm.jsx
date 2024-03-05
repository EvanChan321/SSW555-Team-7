import React, { useState } from "react";

function TestForm({ onSubmit }) {
    const [inputValue, setInputValue] = useState("");

    const handleSubmit = (event) => {
        event.preventDefault();
        onSubmit(inputValue);
    };

    return (
        <form onSubmit={handleSubmit}>
            <label htmlFor="simpleInput">Enter Text:</label>
            <input
                id="simpleInput"
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
            />
            <button type="submit">Submit</button>
        </form>
    );
}

export default TestForm;
