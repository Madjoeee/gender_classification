import './App.css';
import { createRef, useEffect, useState } from 'react';

function App() {

  const models = [
    'VIOLA JONES & CoLBP',
    'GLCM & CoLBP']


  const [message, setMessage] = useState("")
  const [image, setImage] = useState()
  const [isLoading, setLoading] = useState(false)
  const [modelUsed, setModel] = useState("VIOLA JONES & CoLBP")
  const fileRef = createRef()

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true)
    setMessage("Loading...")
    const formData = new FormData();
    formData.append(
      'files', fileRef.current.files[0]
    )
    // setImage(URL.createObjectURL(event.target.files[0]))
    // setMessage(`Selected File ${fileRef.current.files[0].name}`)
    try {
      fetch(
        'http://127.0.0.1:8080/api/' + models.indexOf(modelUsed), {
        method: 'POST',
        body: formData
      }
      )
        .then((response) => response.json())
        .then((data) => {
          setMessage(data["message"])
          setLoading(false)
        })
    } catch (error) {
      setMessage(error)
      setLoading(false)
    }
  }

  const handleModelChange = (e) => {
    setModel(e.target.value)
  }

  function handleChange(e) {

    if (e.target.files[0] !== undefined) {
      setImage(URL.createObjectURL(e.target.files[0]))
    } else {
      setImage()
    }
  }


  return (
    <main className="App text-zinc-950 h-screen">
      <div className='pt-12 flex flex-col justify-center gap-y-12'>
        <p className='ml-12 font-semibold text-2xl'>
          Gender Classification Based on Facial Image
        </p>
        <div className='flex justify-center'>
          <div className='h-[42rem] w-[62rem] bg-white rounded-lg shadow shadow-black/40 flex flex-col justify-center items-center gap-y-3'>
            <div className='prediction-wrapper flex flex-col gap-y-2' >
              {
                image !== undefined ? <img src={image} alt='Images' className=' h-[16.75rem] rounded' />
                  : <div className='w-[16.75rem] h-[16.75rem] flex flex-col justify-center rounded'>No Image Chosen</div>
              }
              <p className='font-semibold'>
                {message}
              </p>

            </div>
            <hr className="h-px bg-black w-[40rem]" />
            <div className='dropdown-wrapper border-b-2 border-sky-300 pt-3' >
              <select
                className='border-0 w-full px-3'
                name='model'
                id='model'
                value={modelUsed}
                onChange={handleModelChange}
              >
                {
                  models.map((model, ix) => ((
                    <option key={ix}>
                      {model}
                    </option>
                  )))
                }

              </select>
            </div>


            <form onSubmit={handleSubmit} method="post" className='flex flex-col ' enctype="multipart/form-data">
              <div className='mt-2'>
                <p className='py-3'>
                  Select image to upload:
                </p>
                <input className='flex justify-center' type="file" multiple accept='image/*' name="image" id="image" ref={fileRef} onChange={handleChange} required />

              </div>
              <div>
                <button type="submit" class={`mt-12 ${isLoading ? 'btn-disbled' : 'btn-sky'}`}>
                  Predict Image
                </button>

              </div>
            </form>

          </div>
        </div>

      </div>
      <footer className='flex justify-center'>
        <div className='absolute bottom-2' >
          <p className='text-zinc-500'>
            Yusrian Darus Syifa (GLCM & CoLBP) | Whinar Kukuh Rizky Ardana (Viola Jones & CoLBP) | Tio Dharmawan | M. Arief Hidayat
          </p>
          <p className='text-zinc-500'>
            Computer Science Faculty, Universitas jember, Indonesia
          </p>

        </div>
      </footer>
    </main>

  );
}

export default App;
