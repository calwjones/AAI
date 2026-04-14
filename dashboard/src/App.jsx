import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleImageUpload = (event) => {
    const file = event.target.files[0]
    if (file) {
      setSelectedImage(file)
      // Create preview URL
      setImagePreview(URL.createObjectURL(file))
    }
  }

  const handleExplain = async () => {
    if (!selectedImage) return
    
    setLoading(true)
    
    try {
      const formData = new FormData()
      formData.append('image', selectedImage)
      formData.append('product_id', 123) // Mock product ID
      
      const response = await axios.post('http://localhost:8001/explain', formData)
      setExplanation(response.data)
    } catch (error) {
      console.error('Error getting explanation:', error)
      alert('Failed to get explanation. Is the API running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="App">
      <h1>XAI Explanation Dashboard</h1>
      
      <div className="upload-section">
        <h2>Upload Image</h2>
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleImageUpload}
        />
        
        {imagePreview && (
          <div>
            <img src={imagePreview} alt="Preview" style={{ maxWidth: '300px' }} />
            <button onClick={handleExplain} disabled={loading}>
              {loading ? 'Analyzing...' : 'Explain Prediction'}
            </button>
          </div>
        )}
      </div>

      {explanation && (
        <div className="explanation-section">
          <h2>Explanation</h2>
          <p><strong>Confidence:</strong> {(explanation.confidence * 100).toFixed(1)}%</p>
          <p><strong>Layer Used:</strong> {explanation.layer_used}</p>
          <p>{explanation.explanation}</p>
          
          {explanation.heatmap_base64 && (
            <div>
              <h3>Grad-CAM Heatmap</h3>
              <img 
                src={`data:image/png;base64,${explanation.heatmap_base64}`} 
                alt="Heatmap"
                style={{ maxWidth: '500px' }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App