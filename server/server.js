import express from "express";
import multer from "multer";
import runPythonInference from "./script.js";
import cors from "cors";
import path from "path";

const app = express();
app.use(cors());

const PORT = 3000;


const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, "uploads/"); 
    },
    filename: function (req, file, cb) {
        cb(
            null,
            file.fieldname + "-" + Date.now() + path.extname(file.originalname)
        );
    },
});

const upload = multer({ storage: storage });

app.post(
    "/analyze-eeg",
    upload.fields([
        { name: "testData", maxCount: 1 },
        { name: "testLabel", maxCount: 1 },
    ]),
    async (req, res) => {
        try {
            const files = req.files;
            if (!files.testData || !files.testLabel) {
                throw new Error(
                    "Both test data and test label files are required."
                );
            }

            const testDataPath = files.testData[0].path;
            const testLabelPath = files.testLabel[0].path;

            const result = await runPythonInference(
                testDataPath,
                testLabelPath
            );
            res.json({ inferenceResult: result });
        } catch (error) {
            console.error(error);
            res.status(500).send(
                "Failed to process EEG data: " + error.message
            );
        }
    }
);

app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}`);
});
