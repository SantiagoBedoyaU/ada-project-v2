import { Button, Col, Form, FormProps, Input, Row, Upload, Card, Radio, ConfigProvider, theme } from 'antd'
import { InboxOutlined } from '@ant-design/icons';
import './App.css'
import { ChangeEvent, useState } from 'react';
import { UploadFile, UploadProps } from 'antd/es/upload';

const { Dragger } = Upload
const abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')

type FieldType = {
  strategy: number,
  initial_state: string,
  candidate_system: string,
  future_subsystem: string,
  present_subsystem: string,
};

function App() {
  const [fileList, setFileList] = useState<UploadFile<any>[]>([]);
  const [logs, setLogs] = useState<String[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [initialStateStr, setInitialStateStr] = useState('')
  const [candidateSystemStr, setCandidateSystem] = useState('')
  const [presentSubsystemStr, setPresentSubsystemStr] = useState('')
  const [futureSubsystemStr, setFutureSubsystemStr] = useState('')

  const onFinish: FormProps<FieldType>['onFinish'] = async (values) => {
    if (fileList.length < 2) {
      alert("Please upload at least 2 TPM files")
    }
    setLogs([])
    setIsLoading(true)
    // console.log('Success:', values);
    // console.log('files', fileList)
    const formData = new FormData()
    fileList.forEach(file => {
      formData.append('tpms', file.originFileObj)
    })
    Object.keys(values).forEach(key => {
      formData.append(key, values[key])
    })

    const res = await fetch("http://localhost:8000/solve", {
      method: 'POST',
      body: formData,
      headers: {
        // 'Content-Type': 'multipart/form-data' // No es necesario establecer este encabezado
      },
    })
    if (res.status === 200) {
      const json = await res.json()
      const l = Object.keys(json).map(key => `${key}: ${json[key]}`)
      setLogs(logs => [...logs, ...l])
    }
    setIsLoading(false)
  };

  const onFinishFailed: FormProps<FieldType>['onFinishFailed'] = (errorInfo) => {
    console.log('Failed:', errorInfo);
  };

  const props: UploadProps = {
    fileList: fileList,
    accept: '.csv',
    multiple: true,
    beforeUpload: () => false,
    onChange(info) {
      let newFileList = [...info.fileList]
      setFileList(newFileList)
    },
    onDrop(e) {
      console.log('Dropped files', e.dataTransfer.files);
    },
  };

  const onChangeInitialState = (e: ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value
    const digits = val.split("")
    let string_digit = ''
    for (let i = 0; i < digits.length; i++) {
      if (digits[i] === '1'){
        string_digit+=abc[i]
      }
    }
    setInitialStateStr(string_digit)
  }

  const onChangeCandidateSystem = (e: ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value
    const digits = val.split("")
    let string_digit = ''
    for (let i = 0; i < digits.length; i++) {
      if (digits[i] === '1'){
        string_digit+=abc[i]
      }
    }
    setCandidateSystem(string_digit)
  }

  const onChangePresentSubsystem = (e: ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value
    const digits = val.split("")
    let string_digit = ''
    for (let i = 0; i < digits.length; i++) {
      if (digits[i] === '1'){
        string_digit+=abc[i]
      }
    }
    setPresentSubsystemStr(string_digit)
  }

  const onChangeFutureSubsystem = (e: ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value
    const digits = val.split("")
    let string_digit = ''
    for (let i = 0; i < digits.length; i++) {
      if (digits[i] === '1'){
        string_digit+=abc[i]
      }
    }
    setFutureSubsystemStr(string_digit)
  }

  return (
    <>
      <ConfigProvider
        theme={{
          algorithm: theme.darkAlgorithm
        }}
      >
        <Row justify='center' style={{ width: '90%', margin: '0 auto' }}>
          <Col span={10} style={{ margin: '0 20px' }}>
            <h2>ADA</h2>
            <Form
              name="basic"
              layout='vertical'
              onFinish={onFinish}
              onFinishFailed={onFinishFailed}
              autoComplete="off"
              initialValues={{
                strategy: 1
              }}
            >
              <Form.Item<FieldType>
                label="Strategy"
                name="strategy"
                rules={[{ required: true, message: 'Please input strategy' }]}
              >
                <Radio.Group>
                  <Radio value={1}>Strategy 1</Radio>
                  <Radio value={2}>Strategy 2</Radio>
                  <Radio value={3}>Strategy 3</Radio>
                </Radio.Group>
              </Form.Item>
              <Form.Item<FieldType>
                label="TPMs"
              >
                <Dragger {...props}>
                  <p className="ant-upload-drag-icon">
                    <InboxOutlined />
                  </p>
                  <p className="ant-upload-text">Click or drag file to this area to upload TPM files</p>
                </Dragger>
              </Form.Item>
              <Row justify='start'>
                <Col style={{ margin: '0 10px' }}>
                  <Form.Item<FieldType>
                    label="Initial State"
                    name="initial_state"
                    rules={[{ required: true, message: 'Please input initial state!' }]}
                    extra={initialStateStr}
                  >
                    <Input onChange={onChangeInitialState} />
                  </Form.Item>
                </Col>
                <Col>
                  <Form.Item<FieldType>
                    label="Candidate System"
                    name="candidate_system"
                    rules={[{ required: true, message: 'Please input candidate system!' }]}
                    extra={candidateSystemStr}
                  >
                    <Input onChange={onChangeCandidateSystem} />
                  </Form.Item>
                </Col>
              </Row>

              <Row>
                <Col style={{ margin: '0 10px' }}>
                  <Form.Item<FieldType>
                    label="Present Subsytem"
                    name="present_subsystem"
                    rules={[{ required: true, message: 'Please input present subsystem!' }]}
                    extra={presentSubsystemStr}
                  >
                    <Input onChange={onChangePresentSubsystem} />
                  </Form.Item>
                </Col>
                <Col>
                  <Form.Item<FieldType>
                    label="Future Subsytem"
                    name="future_subsystem"
                    rules={[{ required: true, message: 'Please input future subsystem!' }]}
                    extra={futureSubsystemStr}
                  >
                    <Input onChange={onChangeFutureSubsystem} />
                  </Form.Item>
                </Col>
              </Row>

              <Form.Item label={null}>
                <Button loading={isLoading} type="primary" htmlType="submit" block>
                  Submit
                </Button>
              </Form.Item>
            </Form>
          </Col>
          <Col span={6}>
            <h2>Results</h2>
            <Card style={{ height: '700px', maxHeight: '700px', overflow: 'auto' }}>
              {logs.map((log, idx) => (
                <code style={{ display: 'block' }} key={idx}>{log}</code>
              ))}
            </Card>
          </Col>
        </Row>
      </ConfigProvider>
    </>
  )
}

export default App
