import './App.css'
import FileUploadForm from "./components/FileUploadForm";
import Footer from "./components/Footer";

const App = () => {
    return (
        <div>
            <h1>File Upload for Lie Detector Data</h1>
            <FileUploadForm />
            <Footer />
        </div>
    );
};

export default App;
