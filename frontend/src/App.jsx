import { useState, useEffect } from 'react';
import './index.css';

function App() {
  const [drugs, setDrugs] = useState([]);
  const [drugA, setDrugA] = useState('');
  const [drugB, setDrugB] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('phase2');

  useEffect(() => {
    // Fetch drug list on mount
    fetch('http://localhost:8000/api/drugs')
      .then(res => res.json())
      .then(data => setDrugs(data.drugs || []))
      .catch(err => console.error("Failed to load drugs:", err));
  }, []);

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!drugA || !drugB) {
      setError('Please select two different drugs.');
      return;
    }
    if (drugA === drugB) {
      setError('Please select two different drugs. You selected the same drug twice.');
      return;
    }
    
    setError(null);
    setLoading(true);
    setResults(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ drug_a: drugA, drug_b: drugB })
      });
      
      if (!response.ok) throw new Error("Failed to fetch predictions from AI.");
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-12 max-w-6xl">
      <header className="mb-12 text-center">
        <h1 className="text-5xl font-extrabold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-sky-400">
          FAERS Pharmacovigilance AI
        </h1>
        <p className="text-slate-400 text-lg max-w-2xl mx-auto">
          Predict adverse events and estimate risk severity for drug-drug interactions using a Machine Learning model trained on millions of historical FDA reports.
        </p>
      </header>

      <div className="grid md:grid-cols-3 gap-8">
        
        {/* Left Sidebar: Search Panel */}
        <div className="md:col-span-1">
          <div className="glass-card">
            <h2 className="text-xl font-bold mb-6 text-slate-100 flex items-center">
              <span className="text-violet-500 mr-2">🔍</span> Select Drugs
            </h2>
            
            <form onSubmit={handlePredict} className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-slate-400 mb-2">Drug A</label>
                <select 
                  className="glass-input"
                  value={drugA} 
                  onChange={(e) => setDrugA(e.target.value)}
                >
                  <option value="">Select a drug...</option>
                  {drugs.map(d => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-slate-400 mb-2">Drug B</label>
                <select 
                  className="glass-input"
                  value={drugB} 
                  onChange={(e) => setDrugB(e.target.value)}
                >
                  <option value="">Select a drug...</option>
                  {drugs.map(d => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>

              {error && <div className="text-rose-400 text-sm font-medium bg-rose-500/10 p-3 rounded-lg border border-rose-500/20">{error}</div>}
              
              <button 
                type="submit" 
                disabled={loading || !drugA || !drugB}
                className={`w-full btn-primary flex justify-center items-center ${loading ? 'opacity-75 cursor-not-allowed' : ''}`}
              >
                {loading ? (
                  <span className="animate-pulse">Analyzing Model...</span>
                ) : (
                  <span>Run Analysis 🚀</span>
                )}
              </button>
            </form>
          </div>
        </div>

        {/* Right Main Panel: Results Dashboard */}
        <div className="md:col-span-2">
          {!results && !loading && (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 border-2 border-dashed border-slate-700 rounded-xl p-12 text-center bg-slate-800/20">
              <span className="text-5xl mb-4">🧪</span>
              <h3 className="text-xl font-medium mb-2">Awaiting Input</h3>
              <p>Select two drugs and run the analysis to view historical evidence and AI risk predictions.</p>
            </div>
          )}

          {loading && (
            <div className="h-full flex items-center justify-center">
              <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-violet-500"></div>
            </div>
          )}

          {results && (
            <div className="animate-fade-in-up">
              <h2 className="text-2xl font-bold mb-6 flex items-center">
                Interaction Analysis: 
                <span className="text-emerald-400 ml-2">{results.drug_a}</span> 
                <span className="text-slate-500 mx-2">+</span> 
                <span className="text-emerald-400">{results.drug_b}</span>
              </h2>

              {/* Tabs */}
              <div className="flex space-x-1 mb-6 border-b border-slate-700">
                <button
                  className={`px-4 py-2 font-medium text-sm rounded-t-lg transition-colors ${activeTab === 'phase2' ? 'bg-slate-800/50 text-violet-500 border-b-2 border-violet-500' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/30'}`}
                  onClick={() => setActiveTab('phase2')}
                >
                  🤖 Phase 2: AI Risk Prediction
                </button>
                <button
                  className={`px-4 py-2 font-medium text-sm rounded-t-lg transition-colors ${activeTab === 'phase1' ? 'bg-slate-800/50 text-sky-400 border-b-2 border-sky-400' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/30'}`}
                  onClick={() => setActiveTab('phase1')}
                >
                  📊 Phase 1: Historical Evidence
                </button>
              </div>

              {/* Tab Content */}
              {activeTab === 'phase2' && (
                <div className="glass-card border-l-4 border-l-violet-500">
                  <h3 className="text-xl font-bold text-violet-500 mb-2">AI Predicted Risk Severity</h3>
                  <p className="text-slate-400 text-sm mb-6">
                    The XGBoost model analyzes the historical baseline features and predicts the underlying biological risk score, learning to filter out purely statistical noise like small-sample coincidences.
                  </p>
                  
                  <div className="bg-slate-900/50 rounded-lg p-6 border border-slate-700 flex flex-col items-center justify-center text-center">
                    <div className="text-slate-400 uppercase tracking-wider text-sm font-bold mb-2">Predicted Severity Score</div>
                    <div className="text-6xl font-black text-white mb-4">
                      {results.ai_prediction.score.toFixed(2)}
                    </div>
                    <div>
                      <span className={`risk-badge-${results.ai_prediction.label.split(' ')[0].toLowerCase()}`}>
                        {results.ai_prediction.label}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'phase1' && (
                <div className="glass-card border-l-4 border-l-sky-400">
                  <h3 className="text-xl font-bold text-sky-400 mb-2">Historical FAERS Evidence</h3>
                  <p className="text-slate-400 text-sm mb-6">
                    Top 10 most significant adverse events reported for this specific combination in the FDA database, ranked by PRR risk score.
                  </p>
                  
                  {results.historical_evidence.length > 0 ? (
                    <div className="overflow-x-auto rounded-lg border border-slate-700">
                      <table className="w-full text-left border-collapse">
                        <thead>
                          <tr className="bg-slate-800 text-slate-300 text-sm uppercase tracking-wider">
                            <th className="p-4 font-medium">Adverse Event</th>
                            <th className="p-4 font-medium text-right">Co-occurrences</th>
                            <th className="p-4 font-medium text-right">PRR</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-700/50">
                          {results.historical_evidence.map((row, idx) => (
                            <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                              <td className="p-4 font-medium text-slate-200 capitalize">{row.event.toLowerCase()}</td>
                              <td className="p-4 text-right text-emerald-400 font-mono">{row.co_occurrences}</td>
                              <td className="p-4 text-right text-amber-400 font-mono">{row.PRR.toFixed(2)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="text-center p-8 bg-slate-900/50 rounded-lg border border-slate-700 text-slate-400">
                      No significant historical evidence found for this specific combination in the database.
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
        
      </div>
      
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in-up {
          animation: fadeInUp 0.4s ease-out forwards;
        }
      `}} />
    </div>
  );
}

export default App;
