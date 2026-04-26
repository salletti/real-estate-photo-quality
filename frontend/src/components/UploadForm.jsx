export default function UploadForm({ onSubmit, loading }) {
  function handleSubmit(e) {
    e.preventDefault();
    const file = e.target.image.files[0];
    onSubmit(file);
  }

  return (
    <form onSubmit={handleSubmit} className="form">
      <div className="field">
        <label>Image</label>
        <input name="image" type="file" accept="image/*" required />
      </div>

      <button type="submit" disabled={loading}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>
    </form>
  );
}
