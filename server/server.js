import express from "express";
import runPythonInference from "./script.js";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 3000;

app.post("/analyze-eeg", async (req, res) => {
    try {
        const eegData = req.body; 
        const result = await runPythonInference(eegData);
        res.json({ inferenceResult: result });
    } catch (error) {
        console.error(error);
        res.status(500).send("Failed to process EEG data");
    }
});

app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}`);
});
