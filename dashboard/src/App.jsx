import { useState } from 'react'
import axios from 'axios'
import './style.css'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
 
 
// Mock accuracy data over time
const mockAccuracyData = [
  { date: 'Apr 8', accuracy: 82 },
  { date: 'Apr 9', accuracy: 84 },
  { date: 'Apr 10', accuracy: 86 },
  { date: 'Apr 11', accuracy: 88 },
  { date: 'Apr 12', accuracy: 87 },
  { date: 'Apr 13', accuracy: 89 },
  { date: 'Apr 14', accuracy: 91 },
  { date: 'Apr 15', accuracy: 90 }
]
// mock quality predictions 
const mockPredictions = [
  {
    id: 1,
    timestamp: '2026-04-15 14:23',
    product: 'Tomatoes',
    producer: 'Green Fields Farm',
    grade: 'B',
    color_score: 68.0,
    size_score: 85.0,
    ripeness_score: 72.0,
    confidence: 0.87,
    model_version: 'v1.2'
  },
  {
    id: 2,
    timestamp: '2026-04-15 13:45',
    product: 'Apples',
    producer: 'Orchard Valley',
    grade: 'A',
    color_score: 88.0,
    size_score: 90.0,
    ripeness_score: 85.0,
    confidence: 0.94,
    model_version: 'v1.2'
  },
  {
    id: 3,
    timestamp: '2026-04-15 12:10',
    product: 'Bananas',
    producer: 'Tropical Imports',
    grade: 'C',
    color_score: 58.0,
    size_score: 70.0,
    ripeness_score: 55.0,
    confidence: 0.78,
    model_version: 'v1.2'
  },
  {
    id: 4,
    timestamp: '2026-04-15 11:30',
    product: 'Carrots',
    producer: 'Root & Co',
    grade: 'A',
    color_score: 85.0,
    size_score: 88.0,
    ripeness_score: 80.0,
    confidence: 0.91,
    model_version: 'v1.2'
  },
  {
    id: 5,
    timestamp: '2026-04-15 10:15',
    product: 'Oranges',
    producer: 'Citrus Grove',
    grade: 'B',
    color_score: 75.0,
    size_score: 82.0,
    ripeness_score: 70.0,
    confidence: 0.85,
    model_version: 'v1.2'
  }
]
 
function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [loading, setLoading] = useState(false)
  
  const handleImageUpload = (event) => {
    const file = event.target.files[0]
    if (file) {
      setSelectedImage(file)
      setImagePreview(URL.createObjectURL(file))
      setExplanation(null)
    }
  }
 
  const handleExplain = async () => {
    if (!selectedImage) return
    
    setLoading(true)
    
    try {
      const formData = new FormData()
      formData.append('image', selectedImage)
      formData.append('product_id', 123)
      
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
    <>
      {/* Simplified Navbar */}
      <nav className="navbar">
        <div className="navbar-brand">DESD Marketplace - Quality Analysis</div>
      </nav>
 
      {/* Full width container */}
      
      <div style={{ 
        maxWidth: '1200px', 
        margin: '0 auto',
        padding: '2rem 1.5rem'
      }}>
        {/* Page Header */}
        <div className="dashboard-header">
          <h1>Quality Analysis Dashboard</h1>
        </div>
 
 
        {/* Accuracy Monitoring Chart */}
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ 
            fontSize: '1.1rem', 
            fontWeight: '600', 
            marginBottom: '0.5rem', 
            color: 'var(--text)' 
          }}>
            Model Accuracy Over Time
          </h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--muted)', marginBottom: '1.5rem' }}>
            7-day rolling accuracy on validation dataset
          </p>
          
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockAccuracyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ddd" />
              <XAxis 
                dataKey="date" 
                style={{ fontSize: '0.8rem', fill: 'var(--muted)' }}
              />
              <YAxis 
                domain={[75, 100]}
                style={{ fontSize: '0.8rem', fill: 'var(--muted)' }}
                tickFormatter={(value) => `${value}%`}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius)',
                  fontSize: '0.85rem'
                }}
                formatter={(value) => [`${value}%`, 'Accuracy']}
              />
              <Line 
                type="monotone" 
                dataKey="accuracy" 
                stroke="var(--accent)" 
                strokeWidth={2.5}
                dot={{ fill: 'var(--accent)', r: 4 }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
          
          {/* Alert threshold indicator */}
          <div style={{ 
            marginTop: '1rem', 
            padding: '0.75rem 1rem',
            backgroundColor: '#e8f5eb',
            border: '1px solid #b0d8b8',
            borderRadius: 'var(--radius)',
            fontSize: '0.85rem',
            color: '#1a5c2a'
          }}>
            <strong>Status:</strong> Model performance is healthy. Current accuracy: 90% (above 85% threshold)
          </div>
        </div>
        
        {/* Recent Predictions Table */}
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ 
            fontSize: '1.1rem', 
            fontWeight: '600', 
            marginBottom: '1rem', 
            color: 'var(--text)' 
          }}>
            Recent Quality Assessments
          </h2>
          
          <table className="product-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Product</th>
                <th>Producer</th>
                <th>Grade</th>
                <th>Scores</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {mockPredictions.map(pred => (
                <tr key={pred.id}>
                  <td>{pred.timestamp.split(' ')[1]}</td>
                  <td>{pred.product}</td>
                  <td style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>
                    {pred.producer}
                  </td>
                  <td>
                    <strong style={{ 
                      color: pred.grade === 'A' ? 'var(--accent)' : 
                            pred.grade === 'C' ? '#c0392b' : 'var(--text)' 
                    }}>
                      Grade {pred.grade}
                    </strong>
                  </td>
                  <td style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>
                    C:{pred.color_score.toFixed(0)}% 
                    S:{pred.size_score.toFixed(0)}% 
                    R:{pred.ripeness_score.toFixed(0)}%
                  </td>
                  <td>{(pred.confidence * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
 
        {/* Upload Card */}
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ 
            fontSize: '1.1rem', 
            fontWeight: '600', 
            marginBottom: '1rem',
            color: 'var(--text)',
            textAlign: 'center'
          }}>
            Upload Product Image
          </h2>
          <div style={{ marginBottom: '1rem', textAlign: 'center' }}>
            <label htmlFor="file-upload" className="btn" style={{ 
              cursor: 'pointer',
              display: 'inline-block'
            }}>
              Choose Image
            </label>
            <input 
              id="file-upload"
              type="file" 
              accept="image/*" 
              onChange={handleImageUpload}
              style={{ display: 'none' }}
            />
            {selectedImage && (
              <span style={{ 
                marginLeft: '1rem', 
                color: 'var(--muted)',
                fontSize: '0.9rem'
              }}>
                {selectedImage.name}
              </span>
            )}
          </div>
          
          {imagePreview && (
            <div style={{ textAlign: 'center' }}>
              <img 
                src={imagePreview} 
                alt="Product" 
                style={{ 
                  maxHeight: '400px',
                  maxWidth: '100%',
                  marginBottom: '1rem',
                  borderRadius: 'var(--radius)'
                }}
              />
              <br />
              <button 
                className="btn"
                onClick={handleExplain} 
                disabled={loading}
              >
                {loading ? 'Analyzing...' : 'Analyze Quality'}
              </button>
            </div>
          )}
        </div>
 
        {/* Results */}
        {explanation && (
          <>
            {/* Summary Cards */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '1rem',
              marginBottom: '1.5rem'
            }}>
              <div className="card">
                <div style={{ 
                  fontSize: '0.75rem', 
                  fontWeight: '600',
                  color: '#666',
                  textTransform: 'uppercase',
                  marginBottom: '0.5rem',
                  letterSpacing: '0.5px'
                }}>
                  Classification
                </div>
                <p style={{ 
                  fontSize: '1.4rem', 
                  fontWeight: '600', 
                  color: 'var(--text)',
                  margin: 0
                }}>
                  {explanation.predicted_class}
                </p>
              </div>
              <div className="card">
                <div style={{ 
                  fontSize: '0.75rem', 
                  fontWeight: '600',
                  color: '#666',
                  textTransform: 'uppercase',
                  marginBottom: '0.5rem',
                  letterSpacing: '0.5px'
                }}>
                  Confidence Level
                </div>
                <p style={{ 
                  fontSize: '1.4rem', 
                  fontWeight: '600', 
                  color: 'var(--accent)',
                  margin: 0
                }}>
                  {(explanation.confidence * 100).toFixed(1)}%
                </p>
              </div>
              
              {/* Quality Assessment Card */}
              {explanation.grade && (
                <div className="card">
                  <div style={{ 
                    fontSize: '0.75rem', 
                    fontWeight: '600',
                    color: '#666',
                    textTransform: 'uppercase',
                    marginBottom: '0.5rem',
                    letterSpacing: '0.5px'
                  }}>
                    Quality Assessment
                  </div>
                  <p style={{ 
                    fontSize: '1.4rem', 
                    fontWeight: '600', 
                    color: explanation.grade === 'A' ? 'var(--accent)' : 
                          explanation.grade === 'C' ? '#c0392b' : 'var(--text)',
                    margin: '0 0 0.75rem 0'
                  }}>
                    Grade {explanation.grade}
                  </p>
                  <div style={{ fontSize: '0.85rem', color: 'var(--muted)', lineHeight: '1.6' }}>
                    <div>Color: {explanation.color_score?.toFixed(1)}%</div>
                    <div>Size: {explanation.size_score?.toFixed(1)}%</div>
                    <div>Ripeness: {explanation.ripeness_score?.toFixed(1)}%</div>
                  </div>
                </div>
              )}
              
              <div className="card">
                <div style={{ 
                  fontSize: '0.75rem', 
                  fontWeight: '600',
                  color: '#666',
                  textTransform: 'uppercase',
                  marginBottom: '0.5rem',
                  letterSpacing: '0.5px'
                }}>
                  Analysis Layer
                </div>
                <p style={{ 
                  fontSize: '0.95rem', 
                  color: 'var(--muted)',
                  margin: 0
                }}>
                  {explanation.layer_used}
                </p>
              </div>
            </div>
 
            {/* Explanation */}
            <div className="card" style={{ marginBottom: '1.5rem' }}>
              <h2 style={{ 
                fontSize: '1.1rem', 
                fontWeight: '600', 
                marginBottom: '0.75rem',
                color: 'var(--text)'
              }}>
                Explanation
              </h2>
              <p style={{ color: 'var(--muted)', margin: 0 }}>
                {explanation.explanation}
              </p>
            </div>
 
            {/* Heatmap */}
            {explanation.heatmap_base64 && (
              <div className="card">
                <h2 style={{ 
                  fontSize: '1.1rem', 
                  fontWeight: '600', 
                  marginBottom: '0.5rem',
                  color: 'var(--text)'
                }}>
                  Visual Analysis (Grad-CAM)
                </h2>
                <p style={{ fontSize: '0.85rem', color: 'var(--muted)', marginBottom: '1rem' }}>
                  Highlighted areas show regions that influenced the quality assessment
                </p>
                <div style={{ textAlign: 'center' }}>
                  <img 
                    src={`data:image/png;base64,${explanation.heatmap_base64}`} 
                    alt="Heatmap"
                    style={{ 
                      maxHeight: '500px',
                      maxWidth: '100%',
                      borderRadius: 'var(--radius)'
                    }}
                  />
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </>
  )
}
 
export default App